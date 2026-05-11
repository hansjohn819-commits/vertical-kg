"""M1 ingest — raw document → initial graph (§16.17 design).

The §16.17 cutover replaces the previous "ingest one big text blob" flow
with a per-page, two-pass, stateful pipeline that mirrors Microsoft
GraphRAG's chunk → entity → relationship → community design (without
the community layer; that's out of scope until §16.2).

Top-level entry: ``ingest_document(storage, vector_store, text_units,
ontology_path, raw_doc_id, pages)``. ``pages`` is a list of page-text
strings (single element for non-paginated sources like .txt/.md).

Pipeline:

    PASS 1 — sequential, stateful entity + relationship extraction.
        For each page in order, send (ontology, prior_entities, page_text)
        to the LLM. Returned entities are deduped against prior_entities
        by case-insensitive label match: a hit appends the chunk_id to
        the existing node's text_unit_ids and stashes the partial summary
        for later fusion; a miss creates a fresh node. Edges resolve
        endpoints against prior_entities + new entities; unresolved
        endpoints are dropped (rule 3 of the prompt forbids inferring
        them anyway).

    PASS 2 — relationship-only, full-document canonical entity context.
        For each page, send (full canonical entity list, page text) and
        ask only for relationships missed in PASS 1. Endpoints must
        match the canonical list exactly. Each returned edge carries an
        evidence_quote that we fuzzy-match against the page text — this
        is the anti-hallucination guard (§16.17 decision 11). Skipped on
        single-page documents (PASS 1 already knew the full prior).

    INTRA-DOC FUSE — per-document description merge.
        Any node hit ≥ 2 times in PASS 1 has its summary fused via the
        same _fuse helper M4b uses for cross-document merges, but with
        thinking=False (§16.17.7). This collapses the multiple partial
        summaries the LLM produced for the same entity across pages
        into one canonical description without losing detail.

Boot-time guarantees the new schema fields (`Node.text_unit_ids`,
`Edge.text_unit_ids`, `Edge.evidence_quote`) exist via the storage
backfill (§16.17.1). Old graphs load with empty defaults; re-ingest
populates them on the §16.17.11 migration step.

Failure handling: when the extractor returns content that isn't a
parseable JSON object, the raw response is dumped to
`data/m1_failures/<ts>_<docid>_attempt<N>.txt` and we retry once with
a stronger reformat instruction. Failures are per-page, not
per-document — one bad page doesn't invalidate the rest.
"""

from __future__ import annotations

import json
import re
import string
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.graph.models import DirectProv, Edge, Node
from src.graph.storage import GraphStorage
from src.graph.text_units import TextUnitStore
from src.graph.tokens import (
    INGEST_INPUT_MAX_TOKENS,
    SUMMARY_MAX_TOKENS,
    count_tokens,
)
from src.graph.vector_store import VectorStore
from src.llm.routing import get_client


# ---------------------------------------------------------------------------
# Prompts (§16.17.4)
# ---------------------------------------------------------------------------

PASS1_SYSTEM_PROMPT = """You extract entities and relationships from a single
page of a document, contributing to a knowledge graph that may already
contain entities found on earlier pages of this same document.

Ontology (use these types when applicable; if a type is not in the list
you may still propose it but be conservative):
{ontology}

Hard rules — never violate:
1. Use canonical full names for every entity. Never output pronouns
   ("it", "they") or definite-only references ("the company", "the firm",
   "the agency", "the report", "the project") as entity labels. If a
   sentence only refers to an entity by pronoun or definite-only form
   AND that entity is not in the prior list, omit the mention.
2. If you cannot identify the canonical name from the page text below
   OR from the "Entities already identified" list, omit the entity
   entirely.
3. Only extract relationships where you have identified BOTH source and
   target as canonical entities. Don't infer relationships not stated in
   the text.
4. Match against the "Entities already identified" list first — if a
   pronoun or alias resolves to one of those, use that exact canonical
   name (don't create a duplicate).
5. summary must be a paraphrase derived only from this page's text.
   Do NOT add facts you happen to know from prior training.
6. source_quote and evidence_quote must be verbatim spans from the page
   text — copy them exactly, do not edit, summarize, translate, or
   paraphrase. If no single span supports an entity, set source_quote
   to "".

Entities already identified in earlier pages of this document:
{prior_entities_block}

Process the page in this order. Follow each step before moving to the
next; do not jump ahead. The order substitutes for an internal chain of
thought:

Step 1 — Entity scan. Read the page text once and list every distinct
entity mentioned by canonical name. Skip pronouns and definite-only
references unless you can resolve them against the prior list (rule 4).

Step 2 — Type and summary. For each entity from Step 1, choose the
closest matching type from the ontology, then write a ≤40-word summary
using only facts present on this page (rule 5).

Step 3 — Source quote. For each entity, locate one or two verbatim
sentences from the page that most directly support it. Copy the text
exactly (rule 6); set "" if no single span fits.

Step 4 — Relationship scan. Walk the page text once more looking for
relationships between entities you identified in Steps 1–3. For each
candidate relationship, follow these sub-steps in order:

  4a. Walk through every relation type in the ontology above and ask:
      "Does this relationship fit?" Check the type's `domain` line
      first — if the endpoint types don't match the domain, the type
      doesn't fit, move on.
  4b. If exactly one ontology type fits, use it.
  4c. If multiple ontology types fit, pick the most specific one
      (e.g., CEO_OF over RELATED_TO; HEADQUARTERED_IN over OWNS).
  4d. If NO ontology type fits and the relationship has a clear,
      stateable name in the text, propose a NEW specific type as a
      verb phrase in SCREAMING_SNAKE_CASE (e.g., FUNDED_BY,
      AUTHORED_WITH, AFFILIATED_WITH, PUBLISHED_BY). Do NOT fall
      through to RELATED_TO just because the work of choosing is
      tedious — most real relationships have a name.
  4e. Use RELATED_TO ONLY when the relationship is genuinely too
      vague to name (e.g., "X and Y are mentioned in the same
      paragraph but no clear verb connects them"). RELATED_TO is
      not a default — it's the last resort after 4a-4d.

  Then copy the verbatim evidence_quote that states the relationship
  (rule 3 + rule 6).

Step 5 — Conservative drop. Re-read your draft list. Drop anything
where you are not confident the page text supports it. Better to omit a
fact than to invent one.

Step 6 — Emit the JSON object. No prose, no commentary, no step labels
in the output.

Output STRICT JSON with exactly this shape:
{{
  "entities": [
    {{
      "label": "<short canonical name>",
      "type": "<entity type>",
      "summary": "<concise paraphrase, ≤40 words>",
      "source_quote": "<one or two verbatim sentences from the page text supporting this entity; ≤60 words; copy text exactly>"
    }}
  ],
  "relationships": [
    {{
      "source": "<label from entities>",
      "target": "<label from entities>",
      "type": "<relation type>",
      "evidence_quote": "<verbatim phrase from the page text supporting the relationship>"
    }}
  ]
}}

Output ONLY the JSON object — no markdown fences, no prose, no chain of
thought, no step labels."""


PASS2_SYSTEM_PROMPT = """You find relationships in a single page that may
have been missed in the first extraction pass. You are given the full
list of canonical entities for this entire document. Only return
relationships where BOTH endpoints exactly match an entity in that list.

Canonical entities for this entire document:
{canonical_entities_block}

Relation types you may use (prefer these; propose new ones only if no
existing type fits):
{relation_types_block}

Hard rules:
1. Both source and target must match a canonical entity name from the
   list above (case-insensitive exact match).
2. The relationship must be stated or directly entailed in the page
   text — do not infer beyond the text.
3. evidence_quote must be a verbatim phrase from the page text
   (copy the words exactly, including capitalization and punctuation).
4. Skip any relationship already obvious from the page itself — focus
   on relationships involving entities that may not have been recognized
   on first pass (because they were defined elsewhere in the document
   and only referenced here by alias / pronoun / context).
5. Do NOT add facts you happen to know from prior training. Every
   relationship must be supported by an evidence_quote you copy from
   this page.

Process the page in this order. Follow each step before moving to the
next; do not jump ahead. The order substitutes for an internal chain
of thought:

Step 1 — Mention scan. Read the page text and note which canonical
entities (by exact or aliased name, including pronouns / definite
references that obviously refer to a canonical entity given the
context) are MENTIONED here. Discard mentions that don't resolve to a
canonical entity from the list above.

Step 2 — Pair scan. For each pair of canonical entities both
mentioned on this page, ask: does the page state a relationship
between them?  If yes, draft (source, target, type, evidence_quote).
If no, skip the pair.

Step 3 — Quote check. For each draft relationship, verify
evidence_quote is a verbatim phrase from the page (rule 3). If you
cannot find an exact phrase, drop the relationship — do not invent or
paraphrase.

Step 4 — Conservative drop. Re-read your draft list. Drop anything
where the relationship is not directly stated on this page (rule 2)
or where either endpoint isn't in the canonical list (rule 1).

Step 5 — Emit the JSON array. No prose, no commentary, no step labels
in the output.

Output STRICT JSON: a single array, possibly empty.
[
  {{
    "source": "<canonical name>",
    "target": "<canonical name>",
    "type": "<relation type>",
    "evidence_quote": "<verbatim phrase from page>"
  }}
]

Output ONLY the JSON array — no markdown fences, no prose, no chain of
thought, no step labels. Empty array [] if you find nothing missed."""


RECLASSIFY_SYSTEM_PROMPT = """You re-classify knowledge-graph edges that
were tagged with the generic "RELATED_TO" type during a fast extraction
pass. For each edge you receive, decide its real relation type.

For each edge follow these sub-steps:
  1. Walk through the ontology relation types below and check the
     `domain` line. If the edge's endpoint types do not match the
     domain of a type, that type is not eligible — skip it.
  2. If exactly one ontology type fits and the evidence_quote supports
     it, choose that type.
  3. If multiple ontology types fit, choose the most specific.
  4. If NO ontology type fits but the relationship has a clear,
     stateable name in the evidence_quote, propose a NEW type as a
     verb phrase in SCREAMING_SNAKE_CASE (e.g., FUNDED_BY,
     AUTHORED_WITH, AFFILIATED_WITH, PUBLISHED_BY).
  5. Keep "RELATED_TO" ONLY if the relationship is genuinely too vague
     to name (e.g., the entities are co-mentioned but no clear verb
     connects them). Do not keep RELATED_TO out of laziness.

Ontology relation types:
{relation_types_block}

Output STRICT JSON: an array of decision objects, one per input edge,
in the SAME ORDER as the input. Do not add, drop, or reorder entries.

[
  {{"edge_index": 0, "decision": "<TYPE_NAME>"}}
]

`decision` is the new type name (uppercase verb phrase) or the literal
string "RELATED_TO" to keep the edge as-is. Output ONLY the JSON array
— no markdown fences, no prose, no chain of thought."""


FUSE_SYSTEM_PROMPT = """You consolidate multiple partial descriptions of
the SAME entity (extracted from different pages of one document) into a
single canonical description. The label, type, and identity of the
entity are fixed — your job is to merge the description text without
losing distinct facts.

Reply with STRICT JSON only, these keys:
{
  "summary": "<≤200 words consolidated summary, plain prose>",
  "detail": "<longer-form notes preserving distinct facts; use bullets if helpful>"
}

Output ONLY the JSON object."""


EXTRACTION_RETRY_REMINDER = (
    "Your previous response could not be parsed as JSON. "
    "Output ONLY the JSON value (object or array) — no markdown code "
    "fences, no prose, no explanation, no chain of thought."
)

FAILURE_DUMP_DIR = Path("data") / "m1_failures"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class IngestResult:
    """Aggregate result for one document (which may span many pages)."""

    run_id: str
    raw_doc_id: str
    pages_processed: int
    nodes_added: int
    edges_added: int
    edges_skipped: int
    pass2_edges_added: int = 0
    pass2_edges_dropped: int = 0  # quote sanity check failures
    nodes_fused: int = 0  # multi-hit nodes that triggered intra-doc fuse
    edges_reclassified: int = 0  # RELATED_TO → specific type by classifier
    edges_reclassified_kept: int = 0  # classifier said "keep RELATED_TO"
    duplicate_edges_removed: int = 0  # collapsed by intra-doc dedup
    page_warnings: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# JSON parsing helpers (carried over + generalized for arrays)
# ---------------------------------------------------------------------------


def _clean_json_text(text: str) -> str:
    """Strip the small zoo of garbage local backends sprinkle into JSON."""
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*\n?", "", s)
    s = re.sub(r"\n?```\s*$", "", s)
    s = re.sub(r"<\|[^|>]*\|>", "", s)
    s = re.sub(r'(?m)^(\s*)\|\s*(?=")', r"\1", s)
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    return s


def _parse_json_loose(text: str, *, expect: str = "object"):
    """Parse a JSON object or array. ``expect`` is "object" or "array"; the
    function falls back to a regex-extracted candidate if the strict parse
    fails on surrounding garbage."""
    if not text:
        return None
    cleaned = _clean_json_text(text)
    try:
        parsed = json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        return parsed
    pat = r"\{.*\}" if expect == "object" else r"\[.*\]"
    m = re.search(pat, cleaned, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0), strict=False)
        except json.JSONDecodeError:
            pass
    return None


def _dump_failure(
    *, raw_doc_id: str, run_id: str, attempt: int, pass_label: str,
    page_num: int | None, input_tokens: int, finish_reason: str | None,
    raw_response: str,
) -> Path:
    FAILURE_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = re.sub(r"[^A-Za-z0-9_.\-#]", "_", raw_doc_id)[:80]
    page_tag = f"_p{page_num}" if page_num is not None else ""
    out = FAILURE_DUMP_DIR / f"{ts}_{safe}{page_tag}_{pass_label}_attempt{attempt}.txt"
    out.write_text(
        f"=== M1 extraction failure ===\n"
        f"timestamp: {ts}\n"
        f"raw_doc_id: {raw_doc_id}\n"
        f"page_num: {page_num}\n"
        f"pass: {pass_label}\n"
        f"run_id: {run_id}\n"
        f"attempt: {attempt}\n"
        f"input_tokens: {input_tokens}\n"
        f"output_tokens (estimated): {count_tokens(raw_response)}\n"
        f"finish_reason: {finish_reason}\n"
        f"\n=== Raw LLM response ===\n{raw_response}\n",
        encoding="utf-8",
    )
    return out


# Per-call timeout for M1 extraction LLM calls. The OpenAI SDK default
# (120 s) is too short for a thinking-on local model on a dense page —
# observed real-world failures around the 120 s mark on a 33-page
# kelp-industry report. Bumping to 600 s gives the model headroom to
# extract 20+ entities + relationships per page, while still bounding
# pathological hangs. M1 also catches per-page exceptions so even a
# 600 s timeout on one page no longer kills the whole document.
M1_LLM_TIMEOUT_S = 600


def _llm_with_retry(
    client, *, system: str, user: str, expect: str, raw_doc_id: str,
    run_id: str, page_num: int | None, pass_label: str, thinking: bool = True,
    temperature: float = 0.2,
) -> tuple[object, str | None, str | None]:
    """Call the LLM, parse JSON; on parse failure dump + retry once.
    Returns (parsed_or_None, warning, finish_reason). Network/timeout
    exceptions are NOT caught here — the caller (per-page loop) wraps
    each call so a single page failure doesn't abort the document."""
    base = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    resp = client.chat(
        messages=base, temperature=temperature, thinking=thinking,
        timeout=M1_LLM_TIMEOUT_S,
    )
    content = resp.choices[0].message.content or ""
    finish = getattr(resp.choices[0], "finish_reason", None)
    parsed = _parse_json_loose(content, expect=expect)
    if parsed is not None:
        return parsed, None, finish
    _dump_failure(
        raw_doc_id=raw_doc_id, run_id=run_id, attempt=1, pass_label=pass_label,
        page_num=page_num, input_tokens=count_tokens(user),
        finish_reason=finish, raw_response=content,
    )
    retry = base + [
        {"role": "assistant", "content": content},
        {"role": "user", "content": EXTRACTION_RETRY_REMINDER},
    ]
    resp2 = client.chat(
        messages=retry, temperature=0.4, thinking=thinking,
        timeout=M1_LLM_TIMEOUT_S,
    )
    content2 = resp2.choices[0].message.content or ""
    finish2 = getattr(resp2.choices[0], "finish_reason", None)
    parsed2 = _parse_json_loose(content2, expect=expect)
    if parsed2 is not None:
        return parsed2, "parse_failed_recovered_on_retry", finish2
    _dump_failure(
        raw_doc_id=raw_doc_id, run_id=run_id, attempt=2, pass_label=pass_label,
        page_num=page_num, input_tokens=count_tokens(user),
        finish_reason=finish2, raw_response=content2,
    )
    return None, "parse_failed_after_retry", finish2


# ---------------------------------------------------------------------------
# Prompt building helpers
# ---------------------------------------------------------------------------


def _format_prior_entities(canonical: list[Node]) -> str:
    """Build the "Entities already identified" block. One line per entity:
    `- <label> (<type>) — <summary first 40 words>`. Empty input renders
    as a clean placeholder so the prompt stays readable on page 1."""
    if not canonical:
        return "(none — this is the first page of the document)"
    lines = []
    for n in canonical:
        brief = " ".join(n.summary.split()[:40])
        lines.append(f"- {n.label} ({n.type}) — {brief}")
    return "\n".join(lines)


def _split_ontology(ontology_text: str) -> tuple[str, str]:
    """Return (full_ontology, relation_types_block). The full ontology
    goes into PASS 1; PASS 2 only needs the relation type names + brief
    semantics so the prompt stays compact."""
    if not ontology_text:
        return "(no ontology provided)", "(no relation types defined)"
    rel_idx = ontology_text.find("# Relation Types")
    if rel_idx == -1:
        return ontology_text, "(no relation types section in ontology)"
    relation_block = ontology_text[rel_idx:]
    # Trim past Global Conventions if present — we only want type names.
    glob_idx = relation_block.find("# Global Conventions")
    if glob_idx != -1:
        relation_block = relation_block[:glob_idx]
    return ontology_text, relation_block.strip()


# ---------------------------------------------------------------------------
# Ontology parser — entity types, relation types, and per-relation domains
# ---------------------------------------------------------------------------

# `domain: A × B` — A and B can each be a single type, a parenthesized
# union `(A | B)`, or the literal `any`. The × character is U+00D7; we
# also accept ASCII `x` as a fallback for hand-edited ontologies.
_DOMAIN_LINE_RE = re.compile(r"^\s*domain:\s*(.+)$", re.MULTILINE)
_DOMAIN_SPLIT_RE = re.compile(r"\s*[×x]\s*", re.IGNORECASE)


def _parse_domain_side(side: str) -> set[str]:
    """Convert one side of a `domain: A × B` declaration into a set of
    allowed types. `any` → empty set (sentinel for 'unconstrained').
    `(A | B)` → {A, B}. Single token → {token}."""
    s = side.strip()
    if s.lower() == "any":
        return set()  # sentinel: no constraint
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    parts = [p.strip() for p in re.split(r"\s*\|\s*", s) if p.strip()]
    return set(parts)


_ALIAS_LINE_RE = re.compile(r"^\s*-\s*(\S+)\s*(?:→|->)\s*(\S+)\s*$")


def parse_aliases(ontology_text: str) -> dict[str, str]:
    """Parse the `# Aliases` section of ontology.md.

    Returns ``{alias: canonical}``. M1 ingest rewrites every model-emitted
    relation type matching an alias to its canonical form *before* the
    domain validator and ontology-proposal logger see it, so a model
    typo like ``AFFULIATED_WITH`` collapses cleanly into the registered
    ``AFFILIATED_WITH`` instead of fragmenting the edge-type vocabulary
    or generating a misleading "new type proposed" event.

    Format inside the section (one per line):
        - ALIAS → CANONICAL          # unicode arrow
        - ALIAS -> CANONICAL         # ASCII fallback

    Lines that don't match the pattern are ignored, so `#` comment
    lines inside the section are fine — the section terminates only
    on a *known* top-level heading (``# Global Conventions`` or
    ``# Evolution Log``), not on any `#` line.
    """
    if not ontology_text:
        return {}
    if "# Aliases" not in ontology_text:
        return {}
    block = ontology_text.split("# Aliases", 1)[1]
    # Terminate on the next *named* top-level section, not just any `#`
    # line — the aliases section often carries `#`-prefixed comments.
    end = re.search(
        r"^# (Global Conventions|Evolution Log|Entity Types|Relation Types)\b",
        block, re.MULTILINE,
    )
    if end:
        block = block[: end.start()]
    out: dict[str, str] = {}
    for line in block.splitlines():
        m = _ALIAS_LINE_RE.match(line)
        if m:
            out[m.group(1).strip()] = m.group(2).strip()
    return out


def parse_ontology(ontology_text: str) -> tuple[set[str], set[str], dict[str, tuple[set[str], set[str]]]]:
    """Parse ontology.md → (entity_types, relation_types, domain_map).

    * entity_types: set of `## Type` headings under `# Entity Types`
    * relation_types: set of `## REL_NAME` headings under `# Relation Types`
    * domain_map: {relation_type: (allowed_src_types, allowed_tgt_types)}.
      Empty set on either side means "any" (no constraint on that side).

    The parser is lenient: missing sections, missing domain lines,
    or weird whitespace all degrade to "no constraint" rather than
    raising. The ontology is human-edited and will get sloppy over
    time — refusing to load on a typo is the wrong tradeoff.
    """
    if not ontology_text:
        return set(), set(), {}

    entity_block = ""
    relation_block = ""
    e_idx = ontology_text.find("# Entity Types")
    r_idx = ontology_text.find("# Relation Types")
    g_idx = ontology_text.find("# Global Conventions")

    if e_idx != -1:
        end = r_idx if r_idx > e_idx else (g_idx if g_idx > e_idx else len(ontology_text))
        entity_block = ontology_text[e_idx:end]
    if r_idx != -1:
        end = g_idx if g_idx > r_idx else len(ontology_text)
        relation_block = ontology_text[r_idx:end]

    entity_types = set(re.findall(r"^##\s+(\S.+?)\s*$", entity_block, re.MULTILINE))
    relation_types = set(re.findall(r"^##\s+(\S.+?)\s*$", relation_block, re.MULTILINE))

    # Parse per-relation domains. Walk each `## NAME ... domain: ...`
    # block. Use the next `## ` heading or end-of-block as terminator.
    domain_map: dict[str, tuple[set[str], set[str]]] = {}
    rel_blocks = re.split(r"^##\s+", relation_block, flags=re.MULTILINE)
    for blk in rel_blocks[1:]:  # [0] is preamble before the first ##
        name_match = re.match(r"(\S.+?)\s*$", blk, re.MULTILINE)
        if not name_match:
            continue
        rel_name = name_match.group(1).strip()
        dom_match = _DOMAIN_LINE_RE.search(blk)
        if not dom_match:
            domain_map[rel_name] = (set(), set())  # no constraint
            continue
        sides = _DOMAIN_SPLIT_RE.split(dom_match.group(1).strip(), maxsplit=1)
        if len(sides) != 2:
            domain_map[rel_name] = (set(), set())
            continue
        src_set = _parse_domain_side(sides[0])
        tgt_set = _parse_domain_side(sides[1])
        domain_map[rel_name] = (src_set, tgt_set)

    return entity_types, relation_types, domain_map


def _domain_ok(
    rel_type: str, src_type: str, tgt_type: str,
    domain_map: dict[str, tuple[set[str], set[str]]],
) -> bool:
    """True iff this edge satisfies the relation's domain constraint.
    Unknown rel_type returns True — caller logs it as a proposal but
    doesn't block (see 'propose new type' rule in PASS 1 prompt).
    """
    if rel_type not in domain_map:
        return True
    src_set, tgt_set = domain_map[rel_type]
    if src_set and src_type not in src_set:
        return False
    if tgt_set and tgt_type not in tgt_set:
        return False
    return True


def _classify_edge_type(
    proposed_type: str,
    src_type: str, src_label: str,
    tgt_type: str, tgt_label: str,
    evidence_quote: str,
    ontology_relation_types: set[str],
    domain_map: dict[str, tuple[set[str], set[str]]],
    log_event,  # callable
    *,
    raw_doc_id: str, run_id: str, page_num: int | None, pass_label: str,
    alias_map: dict[str, str] | None = None,
) -> str:
    """Validate an LLM-proposed edge type against the ontology.

    Returns the type that should actually be written on the edge:
      * unchanged if the type is in the ontology AND the domain matches;
      * unchanged BUT logged as `relation_type_proposed` if the model
        invented a type not in the ontology (§16.2 evolution signal);
      * downgraded to `"RELATED_TO"` AND logged as `domain_mismatch`
        if the type is registered but used outside its declared domain
        (§16.17 problem 4 fix).

    If ``alias_map`` is provided and ``proposed_type`` matches a key,
    the alias is resolved to its canonical form *before* any of the
    above logic runs — and a single ``kind=alias_resolved`` audit
    event records the rewrite. This keeps model typos like
    ``AFFULIATED_WITH`` from masquerading as genuine new-type
    proposals (§16.17 round-2 evolution).

    All logging goes through `log_event` (so this helper stays pure
    w.r.t. the project_guide.md / log.md indirection). Truncated quote
    keeps individual log lines bounded.
    """
    if alias_map and proposed_type in alias_map:
        canonical = alias_map[proposed_type]
        log_event({
            "kind": "alias_resolved",
            "alias": proposed_type,
            "canonical": canonical,
            "src_type": src_type, "src_label": src_label,
            "tgt_type": tgt_type, "tgt_label": tgt_label,
            "raw_doc_id": raw_doc_id, "run_id": run_id,
            "page_num": page_num, "pass": pass_label,
            "summary": f"alias rewrite: {proposed_type} → {canonical}",
        })
        proposed_type = canonical
    if proposed_type not in ontology_relation_types:
        log_event({
            "kind": "ontology_proposal",
            "subkind": "relation_type_proposed",
            "proposed_type": proposed_type,
            "src_type": src_type, "src_label": src_label,
            "tgt_type": tgt_type, "tgt_label": tgt_label,
            "evidence_quote": (evidence_quote or "")[:240],
            "raw_doc_id": raw_doc_id, "run_id": run_id,
            "page_num": page_num, "pass": pass_label,
            "summary": f"new relation type proposed: {proposed_type} ({src_type}→{tgt_type})",
        })
        return proposed_type
    if not _domain_ok(proposed_type, src_type, tgt_type, domain_map):
        log_event({
            "kind": "ontology_proposal",
            "subkind": "domain_mismatch",
            "relation_type": proposed_type,
            "src_type": src_type, "src_label": src_label,
            "tgt_type": tgt_type, "tgt_label": tgt_label,
            "evidence_quote": (evidence_quote or "")[:240],
            "raw_doc_id": raw_doc_id, "run_id": run_id,
            "page_num": page_num, "pass": pass_label,
            "summary": (
                f"domain mismatch: {proposed_type} "
                f"{src_type}→{tgt_type} (downgraded to RELATED_TO)"
            ),
        })
        return "RELATED_TO"
    return proposed_type


def _maybe_log_entity_proposal(
    etype: str, label: str,
    ontology_entity_types: set[str],
    log_event,
    *,
    raw_doc_id: str, run_id: str, page_num: int | None,
) -> None:
    """If the model emits an entity type not in the ontology, record it
    as an evolution signal. The type is kept on the node either way —
    M1 entity types have always been free strings (§16.2). Logging
    just makes the proposal observable so a future review tool can
    aggregate counts."""
    if etype in ontology_entity_types:
        return
    log_event({
        "kind": "ontology_proposal",
        "subkind": "entity_type_proposed",
        "proposed_type": etype,
        "label": label,
        "raw_doc_id": raw_doc_id, "run_id": run_id,
        "page_num": page_num,
        "summary": f"new entity type proposed: {etype} ({label})",
    })


# ---------------------------------------------------------------------------
# Quote sanity check (§16.17 decision 11: fuzzy substring match)
# ---------------------------------------------------------------------------


_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})


def _normalize_for_match(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace. Used for the
    PASS 2 evidence_quote sanity check — LLM-emitted quotes routinely
    differ from the page text by a stray space or punctuation, so we
    compare on a normalized form."""
    if not text:
        return ""
    s = text.lower().translate(_PUNCT_TABLE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _quote_matches(quote: str, page_text: str) -> bool:
    if not quote:
        return False
    nq = _normalize_for_match(quote)
    if len(nq) < 8:  # pathologically short — likely garbage
        return False
    np = _normalize_for_match(page_text)
    return nq in np


# ---------------------------------------------------------------------------
# Description fusion (intra-doc, §16.17.3)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Post-pass relation reclassifier (§16.17 problem 3D, 2026-05-09)
# ---------------------------------------------------------------------------

# Number of RELATED_TO edges to send in a single classifier LLM call.
# 10 keeps each prompt small enough to execute fast (thinking-off ~2-5 s)
# while amortising the per-call overhead. Tuned blind for now; adjust
# after smoke-testing if classifier latency becomes the long pole.
_RECLASSIFY_BATCH = 10


def _format_edges_for_classifier(edges: list[Edge], storage: GraphStorage) -> str:
    """Render a batch of edges as `Edge N: ...` lines for the prompt.
    Includes endpoint types so the LLM can apply ontology domain
    constraints, plus the verbatim evidence_quote that justifies the
    relationship."""
    lines: list[str] = []
    for i, e in enumerate(edges):
        src = storage.get_node(e.source_id)
        tgt = storage.get_node(e.target_id)
        if src is None or tgt is None:
            lines.append(f'Edge {i}: [missing endpoint, skip] decision: "RELATED_TO"')
            continue
        quote = (e.evidence_quote or "").strip()
        if not quote:
            quote = "(no evidence_quote was captured)"
        lines.append(
            f"Edge {i}: [{src.type}] {src.label!r} -> [{tgt.type}] {tgt.label!r}\n"
            f"  evidence: {quote[:300]}"
        )
    return "\n".join(lines)


def _reclassify_related_to_edges(
    client,
    storage: GraphStorage,
    relation_types_block: str,
    ontology_relation_types: set[str],
    domain_map: dict[str, tuple[set[str], set[str]]],
    log_event,
    *,
    raw_doc_id: str, run_id: str,
    alias_map: dict[str, str] | None = None,
) -> dict:
    """Re-categorize every edge currently typed RELATED_TO.

    Sends edges to the LLM in batches of `_RECLASSIFY_BATCH`. For each
    edge the model returns a decision: an ontology type, a proposed
    new type, or the literal "RELATED_TO" (keep as-is). Decisions are
    validated against the domain map before being applied — a model
    that picks an ontology type whose domain doesn't fit gets caught
    here and the edge stays RELATED_TO (with an `ontology_proposal:
    domain_mismatch` event logged for audit).

    Returns a stats dict the caller can fold into the ingest summary.
    """
    candidates = [e for e in storage.edges() if e.type == "RELATED_TO"]
    stats = {"reclassified": 0, "kept_related_to": 0, "proposed_new": 0,
             "rejected_domain": 0, "batches": 0, "errors": 0}
    if not candidates:
        return stats

    system = RECLASSIFY_SYSTEM_PROMPT.format(relation_types_block=relation_types_block)

    for batch_start in range(0, len(candidates), _RECLASSIFY_BATCH):
        batch = candidates[batch_start: batch_start + _RECLASSIFY_BATCH]
        user = _format_edges_for_classifier(batch, storage)
        stats["batches"] += 1
        try:
            parsed, _warning, _finish = _llm_with_retry(
                client, system=system, user=user, expect="array",
                raw_doc_id=raw_doc_id, run_id=run_id, page_num=None,
                pass_label="reclassify", thinking=False,
            )
        except Exception as exc:
            stats["errors"] += 1
            log_event({
                "kind": "ingest_reclassify_error",
                "run_id": run_id, "raw_doc_id": raw_doc_id,
                "summary": (
                    f"reclassify batch {stats['batches']} failed "
                    f"{type(exc).__name__}: {str(exc)[:120]}"
                ),
            })
            continue
        if not isinstance(parsed, list):
            stats["errors"] += 1
            continue

        # Index decisions by edge_index and apply.
        decisions = {}
        for d in parsed:
            if not isinstance(d, dict):
                continue
            try:
                idx = int(d.get("edge_index"))
            except (TypeError, ValueError):
                continue
            decision = str(d.get("decision", "")).strip()
            if decision:
                decisions[idx] = decision

        for i, e in enumerate(batch):
            new_type = decisions.get(i, "RELATED_TO")
            if new_type == "RELATED_TO":
                stats["kept_related_to"] += 1
                continue
            src = storage.get_node(e.source_id)
            tgt = storage.get_node(e.target_id)
            if src is None or tgt is None:
                stats["kept_related_to"] += 1
                continue
            # Run the same validator the PASS 1/2 edge writer uses.
            # Side effect: emits ontology_proposal:relation_type_proposed
            # OR ontology_proposal:domain_mismatch as appropriate.
            final_type = _classify_edge_type(
                new_type, src.type, src.label, tgt.type, tgt.label,
                e.evidence_quote, ontology_relation_types, domain_map,
                log_event,
                raw_doc_id=raw_doc_id, run_id=run_id, page_num=None,
                pass_label="reclassify",
                alias_map=alias_map,
            )
            if final_type == "RELATED_TO":
                # Validator downgraded — keep edge as RELATED_TO. Counts
                # as a domain rejection because the model picked a real
                # type that didn't fit.
                stats["rejected_domain"] += 1
                continue
            e.type = final_type
            stats["reclassified"] += 1
            if final_type not in ontology_relation_types:
                stats["proposed_new"] += 1

    log_event({
        "kind": "ingest_reclassify_done",
        "run_id": run_id, "raw_doc_id": raw_doc_id,
        "summary": (
            f"reclassified {stats['reclassified']}/{len(candidates)} RELATED_TO edges "
            f"(kept {stats['kept_related_to']}, proposed_new {stats['proposed_new']}, "
            f"domain_rejected {stats['rejected_domain']}, errors {stats['errors']})"
        ),
        **stats,
        "candidates_total": len(candidates),
    })
    return stats


def _fuse_partial_descriptions(
    client, label: str, summaries: list[str], details: list[str],
) -> tuple[str, str]:
    """Collapse multiple partial summaries/details into one canonical pair.
    Calls the local LLM with thinking=False (§16.17.7) — fusion is a
    text-merge task, not a reasoning task.
    """
    user = (
        f"Entity label: {label}\n\n"
        + "Partial summaries collected from different pages of the same "
        + "document:\n"
        + "\n\n".join(f"[partial #{i+1}] {s}" for i, s in enumerate(summaries))
        + "\n\nAdditional detail fragments (may be empty):\n"
        + "\n\n".join(f"[detail #{i+1}] {d}" for i, d in enumerate(details) if d)
    )
    resp = client.chat(
        messages=[
            {"role": "system", "content": FUSE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        thinking=False,
    )
    content = resp.choices[0].message.content or ""
    parsed = _parse_json_loose(content, expect="object") or {}
    fused_summary = str(parsed.get("summary", "")).strip()
    fused_detail = str(parsed.get("detail", "")).strip()
    if not fused_summary:
        fused_summary = summaries[0]
    if count_tokens(fused_summary) > SUMMARY_MAX_TOKENS:
        fused_summary = fused_summary[: SUMMARY_MAX_TOKENS * 4]
    return fused_summary, fused_detail


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def ingest_document(
    *,
    storage: GraphStorage,
    vector_store: VectorStore | None,
    text_units: TextUnitStore,
    ontology_path: str | Path | None,
    raw_doc_id: str,
    pages: list[str],
) -> IngestResult:
    """High-level §16.17 ingest. Pages are an ordered list of strings; for
    non-paginated sources (.txt/.md) pass a single-element list.

    Resilience contract (added after the 2026-05-08 timeout incident):
      * Each per-page LLM call is wrapped in try/except; a single page
        failing (timeout, connection, bad JSON after retry) records a
        page warning and moves on instead of aborting the document.
      * After every page in PASS 1 and PASS 2, the three stores
        (text_units, storage, vector_store) are saved to disk. A
        document that crashes on page 28 of 33 keeps pages 1-27 of
        verifiable extraction work.
      * A `log_event({"kind": "ingest_page_done", ...})` line goes to
        log.md per page so progress is visible mid-flight.
    """
    if not pages:
        return IngestResult(
            run_id=f"m1-{uuid.uuid4().hex[:8]}", raw_doc_id=raw_doc_id,
            pages_processed=0, nodes_added=0, edges_added=0, edges_skipped=0,
        )

    # Pre-flight cap. Per-page chunking means single pages are tiny, but a
    # caller might still hand us a giant single-element list (unpaginated
    # txt). Reject early instead of letting the LLM truncate.
    for idx, p in enumerate(pages):
        ptok = count_tokens(p)
        if ptok > INGEST_INPUT_MAX_TOKENS:
            raise ValueError(
                f"Page {idx + 1} is {ptok} tiktoken tokens, exceeds M1 cap "
                f"{INGEST_INPUT_MAX_TOKENS}. Split the source first."
            )

    run_id = f"m1-{uuid.uuid4().hex[:8]}"
    client = get_client("backend")
    ontology_text = ""
    if ontology_path is not None:
        try:
            ontology_text = Path(ontology_path).read_text(encoding="utf-8")
        except Exception:
            ontology_text = ""
    full_ontology, relation_types_block = _split_ontology(ontology_text)
    # §16.17 ontology evolution Phase 1 (2026-05-09): parse domain
    # constraints up front so PASS 1 / PASS 2 / post-classifier can
    # validate edges and emit `kind=ontology_proposal` events for any
    # type or domain that doesn't match the registered ontology.
    ontology_entity_types, ontology_relation_types, ontology_domain_map = (
        parse_ontology(ontology_text)
    )
    # §16.17 round-2 (2026-05-09 evening): alias map collapses model
    # spelling errors into canonical types before validation. Empty
    # dict if no `# Aliases` section exists or it has no entries.
    ontology_aliases = parse_aliases(ontology_text)

    from src.modules.m4_sleep_pass.pass_log import log_event

    def _save_all() -> None:
        """Persist all three stores. Called after every page so a crash
        mid-document loses at most the in-flight page, not all prior
        work. Costs <100 ms at our scale; negligible vs LLM call time."""
        text_units.save()
        storage.save()
        if vector_store is not None:
            vector_store.save()

    # Persist text_units up front. They're real even if extraction fails
    # — losing them means re-paying the PDF parse to recover, and the
    # boot consistency check (post-2026-05-08 fix) no longer deletes
    # unreferenced chunks, so they survive a crashed PASS 1.
    chunk_ids: list[str] = []
    for page_idx, page_text in enumerate(pages):
        chunk_id = uuid.uuid4().hex
        text_units.add({
            "id": chunk_id,
            "text": page_text,
            "raw_doc_id": raw_doc_id,
            "page_num": page_idx + 1,
            "n_tokens": count_tokens(page_text),
            "created_at": _now_iso(),
        })
        chunk_ids.append(chunk_id)
    text_units.save()  # commit chunks before any LLM call
    log_event({
        "kind": "ingest_started",
        "run_id": run_id,
        "raw_doc_id": raw_doc_id,
        "summary": f"{raw_doc_id}: {len(pages)} pages persisted, starting PASS 1",
        "pages_total": len(pages),
    })

    # State carried across PASS 1 pages.
    label_to_node: dict[str, Node] = {}        # casefold(label) -> Node
    partial_summaries: dict[str, list[str]] = {}  # node_id -> list of partial summaries
    partial_details: dict[str, list[str]] = {}    # node_id -> list of partial details
    multi_hit_node_ids: set[str] = set()
    page_warnings: list[dict] = []
    nodes_added = 0
    edges_added = 0
    edges_skipped = 0

    # =====================================================================
    # PASS 1
    # =====================================================================
    canonical_so_far: list[Node] = []  # ordered by first-seen for prompt stability
    for page_idx, (page_text, chunk_id) in enumerate(zip(pages, chunk_ids)):
        page_num = page_idx + 1
        prior_block = _format_prior_entities(canonical_so_far)
        system = PASS1_SYSTEM_PROMPT.format(
            ontology=full_ontology,
            prior_entities_block=prior_block,
        )
        user = (
            f"Document: {raw_doc_id}\n"
            f"Page: {page_num} / {len(pages)}\n\n"
            f"{page_text}"
        )
        nodes_before = nodes_added
        edges_before = edges_added
        try:
            # §16.17.7 (revised 2026-05-09): M1 PASS 1 runs thinking=False.
            # Two thinking-on passes per document × 33+ pages × dense reports
            # was bumping us into 30-60 min ingests and 120 s per-call timeouts.
            # Compensation: the system prompt now includes an explicit
            # 6-step procedure that substitutes for chain-of-thought.
            parsed, warning, finish = _llm_with_retry(
                client, system=system, user=user, expect="object",
                raw_doc_id=raw_doc_id, run_id=run_id, page_num=page_num,
                pass_label="pass1", thinking=False,
            )
        except Exception as exc:
            page_warnings.append({
                "page_num": page_num, "pass": "pass1",
                "warning": "llm_call_exception",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:240],
            })
            log_event({
                "kind": "ingest_page_failed",
                "run_id": run_id, "raw_doc_id": raw_doc_id,
                "summary": f"PASS1 page {page_num}/{len(pages)} {type(exc).__name__}: {str(exc)[:120]}",
                "page_num": page_num, "pass": "pass1",
            })
            _save_all()
            continue
        if warning:
            page_warnings.append({
                "page_num": page_num, "pass": "pass1", "warning": warning,
                "finish_reason": finish,
            })
        if parsed is None:
            log_event({
                "kind": "ingest_page_done",
                "run_id": run_id, "raw_doc_id": raw_doc_id,
                "summary": f"PASS1 page {page_num}/{len(pages)} parse_failed nodes=+0 edges=+0",
                "page_num": page_num, "pass": "pass1",
                "nodes_added_this_page": 0, "edges_added_this_page": 0,
                "warning": warning,
            })
            _save_all()
            continue

        # Resolve entities against prior; new ones become Nodes.
        page_label_to_node: dict[str, Node] = {}
        for entity in parsed.get("entities", []) or []:
            label = str(entity.get("label", "")).strip()
            if not label:
                continue
            etype = str(entity.get("type", "Entity")).strip() or "Entity"
            summary = str(entity.get("summary", "")).strip()
            if count_tokens(summary) > SUMMARY_MAX_TOKENS:
                summary = summary[: SUMMARY_MAX_TOKENS * 4]
            source_quote = str(entity.get("source_quote", "")).strip() or None

            key = label.casefold()
            existing = label_to_node.get(key)
            if existing is not None:
                # Hit — append chunk_id, stash partial for fuse.
                if chunk_id not in existing.text_unit_ids:
                    existing.text_unit_ids = list(existing.text_unit_ids) + [chunk_id]
                if summary:
                    partial_summaries.setdefault(existing.id, []).append(summary)
                multi_hit_node_ids.add(existing.id)
                page_label_to_node[label] = existing
                continue

            # Miss — create.
            _maybe_log_entity_proposal(
                etype, label, ontology_entity_types, log_event,
                raw_doc_id=raw_doc_id, run_id=run_id, page_num=page_num,
            )
            node = Node(
                type=etype,
                label=label,
                summary=summary,
                provenance=DirectProv(
                    raw_doc_id=raw_doc_id,
                    line_or_span=source_quote,
                    extraction_run_id=run_id,
                ),
                text_unit_ids=[chunk_id],
            )
            storage.add_node(node)
            if vector_store is not None:
                from src.graph.retrieval import encode_node
                vector_store.add(node.id, encode_node(node))
            label_to_node[key] = node
            partial_summaries[node.id] = [summary] if summary else []
            page_label_to_node[label] = node
            canonical_so_far.append(node)
            nodes_added += 1

        # Resolve edges. Endpoints can match either page-local or
        # document-wide entries (page-local was just registered above
        # so it's in label_to_node already).
        for rel in parsed.get("relationships", []) or []:
            src_label = str(rel.get("source", "")).strip()
            tgt_label = str(rel.get("target", "")).strip()
            if not src_label or not tgt_label:
                edges_skipped += 1
                continue
            src = label_to_node.get(src_label.casefold())
            tgt = label_to_node.get(tgt_label.casefold())
            if not src or not tgt:
                edges_skipped += 1
                continue
            evidence_quote = str(rel.get("evidence_quote", "")).strip()
            proposed_type = str(rel.get("type", "RELATED_TO")).strip() or "RELATED_TO"
            final_type = _classify_edge_type(
                proposed_type, src.type, src.label, tgt.type, tgt.label,
                evidence_quote, ontology_relation_types, ontology_domain_map,
                log_event,
                raw_doc_id=raw_doc_id, run_id=run_id, page_num=page_num,
                pass_label="pass1",
                alias_map=ontology_aliases,
            )
            edge = Edge(
                source_id=src.id,
                target_id=tgt.id,
                type=final_type,
                provenance=DirectProv(
                    raw_doc_id=raw_doc_id,
                    extraction_run_id=run_id,
                ),
                text_unit_ids=[chunk_id],
                evidence_quote=evidence_quote,
            )
            storage.add_edge(edge)
            edges_added += 1

        log_event({
            "kind": "ingest_page_done",
            "run_id": run_id, "raw_doc_id": raw_doc_id,
            "summary": (
                f"PASS1 page {page_num}/{len(pages)} "
                f"nodes=+{nodes_added - nodes_before} "
                f"edges=+{edges_added - edges_before} "
                f"(running total nodes={nodes_added} edges={edges_added})"
            ),
            "page_num": page_num, "pass": "pass1",
            "nodes_added_this_page": nodes_added - nodes_before,
            "edges_added_this_page": edges_added - edges_before,
        })
        _save_all()

    # =====================================================================
    # PASS 2 — relationship-only, full canonical context
    # =====================================================================
    pass2_added = 0
    pass2_dropped = 0
    if len(pages) > 1 and canonical_so_far:
        canon_block = _format_prior_entities(canonical_so_far)
        canon_lookup = {n.label.casefold(): n for n in canonical_so_far}
        for page_idx, (page_text, chunk_id) in enumerate(zip(pages, chunk_ids)):
            page_num = page_idx + 1
            system = PASS2_SYSTEM_PROMPT.format(
                canonical_entities_block=canon_block,
                relation_types_block=relation_types_block,
            )
            user = page_text
            p2_added_before = pass2_added
            p2_dropped_before = pass2_dropped
            try:
                # §16.17.7 (revised 2026-05-09): same thinking=False as PASS 1.
                # PASS 2 is strictly easier (canonical entity list already
                # known, only relationships to find) so the prompt-driven
                # 5-step procedure carries even more weight here.
                parsed, warning, finish = _llm_with_retry(
                    client, system=system, user=user, expect="array",
                    raw_doc_id=raw_doc_id, run_id=run_id, page_num=page_num,
                    pass_label="pass2", thinking=False,
                )
            except Exception as exc:
                page_warnings.append({
                    "page_num": page_num, "pass": "pass2",
                    "warning": "llm_call_exception",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc)[:240],
                })
                log_event({
                    "kind": "ingest_page_failed",
                    "run_id": run_id, "raw_doc_id": raw_doc_id,
                    "summary": f"PASS2 page {page_num}/{len(pages)} {type(exc).__name__}: {str(exc)[:120]}",
                    "page_num": page_num, "pass": "pass2",
                })
                _save_all()
                continue
            if warning:
                page_warnings.append({
                    "page_num": page_num, "pass": "pass2", "warning": warning,
                    "finish_reason": finish,
                })
            if not isinstance(parsed, list):
                log_event({
                    "kind": "ingest_page_done",
                    "run_id": run_id, "raw_doc_id": raw_doc_id,
                    "summary": f"PASS2 page {page_num}/{len(pages)} parse_failed",
                    "page_num": page_num, "pass": "pass2",
                    "edges_added_this_page": 0,
                })
                _save_all()
                continue
            for rel in parsed:
                if not isinstance(rel, dict):
                    continue
                src_label = str(rel.get("source", "")).strip()
                tgt_label = str(rel.get("target", "")).strip()
                quote = str(rel.get("evidence_quote", "")).strip()
                proposed_type = str(rel.get("type", "RELATED_TO")).strip() or "RELATED_TO"
                src = canon_lookup.get(src_label.casefold())
                tgt = canon_lookup.get(tgt_label.casefold())
                if not src or not tgt:
                    pass2_dropped += 1
                    # Endpoint mismatch — model proposed a relationship between
                    # entities that don't both exist in the canonical list.
                    # Logged so PASS 2 quality can be audited; no graph mutation.
                    log_event({
                        "kind": "ingest_quote_dropped",
                        "subkind": "endpoint_mismatch",
                        "run_id": run_id, "raw_doc_id": raw_doc_id,
                        "page_num": page_num, "pass": "pass2",
                        "src_label": src_label, "tgt_label": tgt_label,
                        "proposed_type": proposed_type,
                        "evidence_quote": quote[:240],
                        "summary": (
                            f"PASS2 dropped: endpoints not in canonical list "
                            f"({src_label!r} -> {tgt_label!r})"
                        ),
                    })
                    continue
                if not _quote_matches(quote, page_text):
                    pass2_dropped += 1
                    # Quote sanity-check failed — model probably hallucinated
                    # the supporting phrase. Log so we can inspect whether the
                    # fuzzy matcher is too strict OR the LLM is fabricating
                    # citations. Either way, dropping the edge is correct;
                    # logging just gives us audit material for §16.3 evaluation.
                    log_event({
                        "kind": "ingest_quote_dropped",
                        "subkind": "quote_not_in_page",
                        "run_id": run_id, "raw_doc_id": raw_doc_id,
                        "page_num": page_num, "pass": "pass2",
                        "src_label": src.label, "tgt_label": tgt.label,
                        "proposed_type": proposed_type,
                        "evidence_quote": quote[:240],
                        "summary": (
                            f"PASS2 dropped: evidence_quote not in page text "
                            f"({src.label!r} -[{proposed_type}]-> {tgt.label!r})"
                        ),
                    })
                    continue
                final_type = _classify_edge_type(
                    proposed_type, src.type, src.label, tgt.type, tgt.label,
                    quote, ontology_relation_types, ontology_domain_map,
                    log_event,
                    raw_doc_id=raw_doc_id, run_id=run_id, page_num=page_num,
                    pass_label="pass2",
                    alias_map=ontology_aliases,
                )
                edge = Edge(
                    source_id=src.id,
                    target_id=tgt.id,
                    type=final_type,
                    provenance=DirectProv(
                        raw_doc_id=raw_doc_id,
                        extraction_run_id=run_id,
                    ),
                    text_unit_ids=[chunk_id],
                    evidence_quote=quote,
                )
                storage.add_edge(edge)
                pass2_added += 1

            log_event({
                "kind": "ingest_page_done",
                "run_id": run_id, "raw_doc_id": raw_doc_id,
                "summary": (
                    f"PASS2 page {page_num}/{len(pages)} "
                    f"edges=+{pass2_added - p2_added_before} "
                    f"dropped={pass2_dropped - p2_dropped_before} "
                    f"(running total p2_edges={pass2_added})"
                ),
                "page_num": page_num, "pass": "pass2",
                "edges_added_this_page": pass2_added - p2_added_before,
                "edges_dropped_this_page": pass2_dropped - p2_dropped_before,
            })
            _save_all()

    # =====================================================================
    # Post-pass classifier — re-categorize RELATED_TO edges (§16.17.3D)
    # =====================================================================
    reclassify_stats = _reclassify_related_to_edges(
        client, storage, relation_types_block,
        ontology_relation_types, ontology_domain_map, log_event,
        raw_doc_id=raw_doc_id, run_id=run_id,
        alias_map=ontology_aliases,
    )
    _save_all()

    # =====================================================================
    # Edge dedup (§16.17 problem 2, 2026-05-09) — collapse duplicates
    # within THIS document only. Cross-doc duplicates stay separate;
    # they're independent confirmations and M4b handles cross-doc merge
    # at sleep-pass time. Within one PDF, repeated headers / author
    # rows / repeated tables produce 4-10× duplicates of the same edge,
    # all with identical evidence_quotes — pure noise.
    # =====================================================================
    dedup_stats = storage.dedupe_edges(
        predicate=lambda e: getattr(e.provenance, "extraction_run_id", None) == run_id,
    )
    log_event({
        "kind": "ingest_dedup_done",
        "run_id": run_id, "raw_doc_id": raw_doc_id,
        "summary": (
            f"dedup merged {dedup_stats['groups_merged']} groups, "
            f"removed {dedup_stats['edges_removed']} duplicate edges"
        ),
        **dedup_stats,
    })
    _save_all()

    # =====================================================================
    # Intra-doc fuse — multi-hit nodes only (≥2 partial summaries)
    # =====================================================================
    nodes_fused = 0
    for nid in multi_hit_node_ids:
        node = storage.get_node(nid)
        if node is None:
            continue
        partials = partial_summaries.get(nid, [])
        # The very first (creation-time) summary is in partials too. Need
        # ≥2 distinct partials to make fusion worthwhile.
        if len(partials) < 2:
            continue
        details = partial_details.get(nid, [])
        try:
            fused_summary, fused_detail = _fuse_partial_descriptions(
                client, node.label, partials, details,
            )
        except Exception:
            continue
        if fused_summary:
            node.summary = fused_summary
        if fused_detail:
            node.detail = fused_detail
        # Re-encode the node so the vector index reflects the fused summary.
        if vector_store is not None:
            from src.graph.retrieval import encode_node
            vector_store.remove(nid)
            vector_store.add(nid, encode_node(node))
        nodes_fused += 1

    # Final save after fuse + the closing audit log.
    _save_all()

    # =====================================================================
    # Audit log
    # =====================================================================
    log_event({
        "kind": "ingest_done",
        "run_id": run_id,
        "raw_doc_id": raw_doc_id,
        "summary": (
            f"{raw_doc_id}: pages={len(pages)}, nodes=+{nodes_added}, "
            f"edges=+{edges_added}+{pass2_added}p2-{dedup_stats['edges_removed']}dup, "
            f"fused={nodes_fused}, "
            f"reclassified={reclassify_stats.get('reclassified', 0)}, "
            f"warnings={len(page_warnings)}"
        ),
        "pages_processed": len(pages),
        "nodes_added": nodes_added,
        "edges_added": edges_added,
        "pass2_edges_added": pass2_added,
        "pass2_edges_dropped": pass2_dropped,
        "edges_skipped": edges_skipped,
        "nodes_fused": nodes_fused,
        "edges_reclassified": reclassify_stats.get("reclassified", 0),
        "edges_reclassified_kept": reclassify_stats.get("kept_related_to", 0),
        "edges_reclassified_proposed_new": reclassify_stats.get("proposed_new", 0),
        "edges_reclassified_domain_rejected": reclassify_stats.get("rejected_domain", 0),
        "duplicate_edges_removed": dedup_stats["edges_removed"],
        "warnings": page_warnings,
    })

    return IngestResult(
        run_id=run_id,
        raw_doc_id=raw_doc_id,
        pages_processed=len(pages),
        nodes_added=nodes_added,
        edges_added=edges_added,
        edges_skipped=edges_skipped,
        pass2_edges_added=pass2_added,
        pass2_edges_dropped=pass2_dropped,
        nodes_fused=nodes_fused,
        edges_reclassified=reclassify_stats.get("reclassified", 0),
        edges_reclassified_kept=reclassify_stats.get("kept_related_to", 0),
        duplicate_edges_removed=dedup_stats["edges_removed"],
        page_warnings=page_warnings,
    )


# ---------------------------------------------------------------------------
# Back-compat shim: callers that still pass a single text blob
# ---------------------------------------------------------------------------


def ingest_raw_text(
    storage: GraphStorage,
    raw_text: str,
    raw_doc_id: str,
    *,
    vector_store: VectorStore | None = None,
    text_units: TextUnitStore | None = None,
    ontology_path: str | Path | None = None,
) -> IngestResult:
    """Back-compat thin wrapper. Wraps ``raw_text`` in a single-element
    pages list and dispatches to :func:`ingest_document`.

    Tests / scripts that don't carry a TextUnitStore can pass None and we
    spin up a throwaway one — it's a small in-memory dict, no I/O until
    save() is called (which the wrapper doesn't do).
    """
    if text_units is None:
        text_units = TextUnitStore(Path("data") / "_throwaway_text_units.json")
    return ingest_document(
        storage=storage,
        vector_store=vector_store,
        text_units=text_units,
        ontology_path=ontology_path,
        raw_doc_id=raw_doc_id,
        pages=[raw_text],
    )
