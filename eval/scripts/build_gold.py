"""Build eval/testset/gold.jsonl from candidates.json + hand-curated questions.

The questions and gold_facts below are hand-written by Claude after reading
candidates.json + the source PDFs. This script just maps each pick to the
underlying chunk_ids / docs / pages so the gold record is reproducible
from data.

Schema:
  question_id        : "q001" .. "q061"
  category           : single_hop | multi_hop | aggregation | oos
  question           : the natural-language question
  gold_chunk_ids     : project chunk_ids (used by internal/external recall)
  gold_evidence      : [{raw_doc_id, page_num}] for page-level recall (all systems)
  gold_doc_titles    : display-title list for citation correctness
  gold_facts         : free-text facts; used by Claude for answer-correctness scoring
  expected_behavior  : "answer" or "refuse"
  meta               : provenance back to candidates idx
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.graph.retrieval import display_title


# ---------- Single-hop picks (candidate_idx -> question, facts) ----------

SINGLE_HOP = {
    1:  ("Which Washington state agency regulates the seaweed aquaculture industry?",
         ["Washington Department of Natural Resources regulates Washington seaweed aquaculture (alongside Washington Department of Fish and Wildlife)."]),
    2:  ("Who is the lead author of the FAO study on assessing marine fishery stocks alongside Felix Marttin?",
         ["Rishi Sharma is the lead author with Felix Marttin (and Marc Taconet)."]),
    4:  ("In which country is the company C-Weed Mwani based?",
         ["C-Weed Mwani is based in Tanzania."]),
    6:  ("Which company is Matt Obee affiliated with?",
         ["Matt Obee is affiliated with Cascadia Seaweed."]),
    11: ("Who co-founded Maine Ocean Farms?",
         ["Eric Oransky and Willy Leathers co-founded Maine Ocean Farms in 2017."]),
    12: ("Who published 'The State of World Fisheries and Aquaculture 2020'?",
         ["FAO published 'The State of World Fisheries and Aquaculture 2020'."]),
    13: ("Who launched Eastern Edge Sea Salt?",
         ["Kelly Hinkle (along with her father, a father-son duo) launched Eastern Edge Sea Salt."]),
    15: ("Who is the President of ASF (Atlantic Salmon Federation) Canada?",
         ["Bill Taylor is the President of ASF (Canada)."]),
    17: ("Does the SSF-LEX (Small-Scale Fisheries legal database) include a profile for Albania?",
         ["Yes — as of April 2024, SSF-LEX includes profiles for Albania (and other countries like Cabo Verde, Tanzania, Namibia)."]),
    18: ("Who commissioned / for whom was the Alaska Seaweed Market Assessment prepared?",
         ["The Alaska Seaweed Market Assessment was prepared for the Alaska Fisheries Development Foundation (AFDF)."]),
    19: ("Which organization funded the Washington Seaweed Aquaculture: Economic Potential Analysis report?",
         ["Builders Vision funded / supported the production of the report."]),
    20: ("Who founded the company Patagonia?",
         ["Yvon Chouinard founded Patagonia."]),
    24: ("Which organization published the SSF Guidelines (Voluntary Guidelines for Securing Sustainable Small-Scale Fisheries)?",
         ["FAO published the SSF Guidelines."]),
    25: ("Who founded the company Hooké?",
         ["Fred Campbell founded Hooké."]),
    26: ("Is China a major exporter of shrimps according to FAO SOFIA 2024?",
         ["Yes — China is a major exporter of shrimps (alongside cephalopods, tilapias, Alaska pollock, mackerels, and tunas)."]),
    27: ("In which country did FISH4ACP develop a value chain analysis for pelagic fishers on Lake Tanganyika?",
         ["United Republic of Tanzania (FISH4ACP studied pelagic fishers on Lake Tanganyika in Tanzania)."]),
    30: ("Who produced the FAO Voluntary Guidelines for Transshipment?",
         ["FAO produced the Voluntary Guidelines for Transshipment."]),
    32: ("Who is the interim CEO of Atlantic Sea Farms (named in late 2025 / early 2026)?",
         ["Mikel Durham is the interim CEO of Atlantic Sea Farms."]),
    35: ("What kind of work has FAO done with national organizations in Chile?",
         ["FAO collaborated with national organizations in Chile to use GIS for marine spatial planning / coastal governance."]),
    38: ("Where (which body of water) did Island Institute Fellow Lorren Ruscetta conduct her work?",
         ["Lorren Ruscetta worked in Casco Bay (Maine)."]),
    39: ("Which agency funded the EAF-Nansen Programme?",
         ["The Norwegian Agency for Development Cooperation funded the EAF-Nansen Programme."]),
    41: ("Was Vietnam a supplier of aquatic animal products to the United States according to FAO SOFIA 2024?",
         ["Yes — Vietnam (Viet Nam) was among the primary suppliers of aquatic animal products to the United States."]),
    45: ("Did Martone R and Gregr IR co-author a 2025 paper cited in the Kelponomics report?",
         ["Yes — Martone R, Gregr IR, and Gregr EJ co-authored a 2025 paper cited in the Kelponomics report."]),
    48: ("Where is sugar kelp (Saccharina latissima) cultured according to the techno-economic analysis?",
         ["Sugar kelp / S. latissima is cultured in North America; it is the most commonly cultured kelp species in NA."]),
    49: ("Is Canada a member of the High Level Panel for a Sustainable Ocean Economy?",
         ["Yes — Canada (along with Mexico) is a member of the High Level Panel for a Sustainable Ocean Economy."]),
    51: ("Did Blue Evolution operate in the North American kelp industry?",
         ["Yes — Blue Evolution operated in the North American kelp industry, but pivoted away from food."]),
    53: ("What happened to AKUA in 2024?",
         ["AKUA closed in 2024, highlighting the difficulty of sustaining kelp-based food businesses."]),
    56: ("In which country was tilapia included in school meals?",
         ["Tilapia was included in school meals in Guatemala."]),
    58: ("Which firm audits ASF (Canada)'s financial statements?",
         ["KPMG LLP audits the financial statements of ASF (Canada) (and ASF U.S.)."]),
    33: ("Who co-founded Maine Ocean Farms in 2017 alongside Eric Oransky?",
         ["Willy Leathers co-founded Maine Ocean Farms in 2017 with Eric Oransky."]),
}


# ---------- Multi-hop picks (candidate_idx -> question, facts) ----------

MULTI_HOP = {
    0: ("Through what location are The Maine Aquaculture Association and the researcher C. Brayden both connected?",
        ["Both are linked through Maine — The Maine Aquaculture Association is located in Maine, and Brayden, C. focuses geographically on Maine."]),
    1: ("Which research group produced the Alaska Seaweed Market Assessment that was prepared for the Alaska Fisheries Development Foundation?",
        ["McKinley Research Group produced the Alaska Seaweed Market Assessment, which was prepared for / published for the Alaska Fisheries Development Foundation (AFDF)."]),
    2: ("Has FAO, which studies U.S. fisheries, also published 'The State of World Fisheries and Aquaculture' report?",
        ["Yes — FAO publishes 'The State of World Fisheries and Aquaculture' (multiple years), and FAO also studies U.S. fisheries / kelp aquaculture sector."]),
    3: ("Are R. Fujita and M. Stekoll connected as co-authors through a common collaborator? If so, who?",
        ["Yes — both have co-authored with C. Yarish (Fujita with Yarish in one paper; Stekoll with Yarish in another)."]),
    4: ("Has C. Yarish co-authored with researchers Kim, J.-K. and Kim, J. (according to the cited papers)?",
        ["Yes — C. Yarish co-authored 2019 papers with both Kim, J.-K. (Kim, Stekoll, & Yarish 2019) and Kim, J."]),
    5: ("Has FAO conducted suitability or sector studies in both the United Arab Emirates and the United States?",
        ["Yes — FAO conducted GIS suitability analysis with the UAE for offshore aquaculture, and FAO covers the U.S. kelp aquaculture sector as part of its global studies."]),
    6: ("Are both researcher Xue W and the Island Institute active in Maine?",
        ["Yes — Xue W studied public perceptions of seaweed aquaculture in Maine (2025), and the Island Institute operates / creates impact along Maine's coast."]),
    7: ("Sugar kelp is grown primarily in a state where Atlantic Sea Farms is also located. Which state?",
        ["Maine — sugar kelp (S. latissima) is primarily grown in Maine, and Atlantic Sea Farms is also based in Maine."]),
    8: ("FocusMaine and Atlantic Sea Farms are both connected to which U.S. state?",
        ["Maine — both FocusMaine and Atlantic Sea Farms are based in / operate in Maine."]),
    10: ("Is Mount Desert Island located in the same state where Xue W conducted her seaweed aquaculture research?",
         ["Yes — Mount Desert Island is in Maine, and Xue W studied seaweed aquaculture in Maine."]),
    11: ("What is the relationship chain connecting McKinley Research Group to the Alaska Fisheries Development Foundation?",
         ["McKinley Research Group produced the Alaska Seaweed Market Assessment, which was prepared for / published by the Alaska Fisheries Development Foundation."]),
}


# ---------- Aggregation picks (hub_idx -> question, gold facts list) ----------
# For aggregation, gold_chunk_ids = union of ALL chunks attached to the hub
# node + its neighbors (since the answer requires evidence from many sources).
# That's coarse, but page-level recall is the more meaningful metric here.

AGGREGATION = [
    {
        "hub_idx": 0,  # FAO
        "question": "List at least 4 publications, programmes, or guidelines produced by FAO that are mentioned in the corpus.",
        "facts": [
            "The State of World Fisheries and Aquaculture (multiple years: 2020, 2022, 2024)",
            "FAO Voluntary Guidelines for Transshipment",
            "Blue Transformation Roadmap",
            "Code of Conduct for Responsible Fisheries (CCRF)",
            "FishStat / FishStatJ",
            "EAF-Nansen Programme",
            "FAO Strategic Framework 2022-2031",
            "FAO Science and Innovation Strategy",
            "FAO International Guidelines on Bycatch Management",
            "Illuminating Hidden Harvests (with Duke and WorldFish)",
        ],
        "neighbor_filter": ("Product",),  # restrict gold neighbor chunks to Product type
    },
    {
        "hub_idx": 2,  # Alaska
        "question": "List several organizations, foundations, or companies active in or connected to Alaska's seaweed/kelp sector.",
        "facts": [
            "Alaska Fisheries Development Foundation (AFDF)",
            "Macro Oceans (operates in Alaska / NA kelp industry)",
            "Other organizations connected to Alaska based on the corpus",
        ],
        "neighbor_filter": ("Organization", "Company"),
    },
    {
        "hub_idx": 3,  # GreenWave
        "question": "Where (which geographic locations or regions) is GreenWave associated with according to the corpus?",
        "facts": [
            "Multiple coastal regions / states in North America (per the State of the Kelp Industry Report and others)",
        ],
        "neighbor_filter": ("Location",),
    },
    {
        "hub_idx": 4,  # Gulf of Maine
        "question": "Name organizations connected to the Gulf of Maine according to the corpus.",
        "facts": ["Multiple organizations linked to the Gulf of Maine ecosystem and industry."],
        "neighbor_filter": ("Organization",),
    },
    {
        "hub_idx": 5,  # North American kelp industry
        "question": "Name several companies operating in the North American kelp industry.",
        "facts": [
            "Atlantic Sea Farms",
            "Cascadia Seaweed",
            "Macro Oceans",
            "Cold Current Kelp",
            "Marine Biologics",
            "Blue Evolution",
            "AKUA",
            "GreenWave",
        ],
        "neighbor_filter": ("Company",),
    },
    {
        "hub_idx": 6,  # ASF (CANADA)
        "question": "List several people / staff associated with ASF (Atlantic Salmon Federation) Canada.",
        "facts": [
            "Bill Taylor (President / Président)",
            "Other staff and board members named in the ASF Impact Report 2024",
        ],
        "neighbor_filter": ("Person",),
    },
    {
        "hub_idx": 9,  # Aquatic animal products
        "question": "Name countries that export aquatic animal products according to FAO SOFIA 2024.",
        "facts": [
            "China (major exporter of shrimps, cephalopods, tilapias, Alaska pollock, mackerels, tunas)",
            "Spain (cephalopods)",
            "Maldives (>30% of merchandise trade value)",
            "Vietnam (supplier to USA)",
            "Italy",
            "Japan",
        ],
        "neighbor_filter": ("Location",),
    },
    {
        "hub_idx": 10,  # Island Institute
        "question": "Name people associated with the Island Institute (in Maine).",
        "facts": [
            "Lorren Ruscetta (Island Institute Fellow)",
            "Other fellows / staff named in the 2025 Impact Report",
        ],
        "neighbor_filter": ("Person",),
    },
    {
        "hub_idx": 11,  # Maine
        "question": "Name companies and organizations located in or operating in Maine according to the corpus.",
        "facts": [
            "Atlantic Sea Farms",
            "Maine Ocean Farms",
            "Eastern Edge Sea Salt",
            "FocusMaine",
            "The Maine Aquaculture Association",
            "Maine Aquaculture Innovation Center",
            "McKinley Research Group",
            "Island Institute",
        ],
        "neighbor_filter": ("Company", "Organization"),
    },
    {
        "hub_idx": 14,  # British Columbia
        "question": "List several organizations or companies operating in British Columbia in the kelp / aquaculture sector.",
        "facts": [
            "Cascadia Seaweed (based in BC area per Washington report)",
            "Other BC-based organizations per the corpus",
        ],
        "neighbor_filter": ("Organization", "Company"),
    },
]


# ---------- Out-of-scope (handwritten, no gold evidence) ----------

OOS = [
    "What is the current price of Bitcoin?",
    "Who is the current Prime Minister of Australia?",
    "What's the weather forecast for Tokyo tomorrow?",
    "Write a Python function that reverses a singly linked list.",
    "What is the chemical formula for caffeine?",
    "Recommend a good Italian restaurant in Boston.",
    "What was the score of the most recent FIFA World Cup final?",
    "Translate 'hello, how are you?' into Russian.",
    "What is the recipe for traditional spaghetti carbonara?",
    "What is the capital city of Mongolia?",
]


def main():
    cand_path = ROOT / "eval" / "testset" / "candidates.json"
    with cand_path.open("r", encoding="utf-8") as f:
        cands = json.load(f)

    sh = cands["single_hop_edges"]
    mh = cands["multi_hop_bridges"]
    hubs = cands["hub_nodes_for_aggregation"]

    out_lines = []
    qid = 0

    def _gold_evidence_from_chunks(chunk_ids: list[str]) -> list[dict]:
        # Pull from cand chunks not directly available — we need actual text_units.
        # Read from production text_units once.
        return [{"chunk_id": c} for c in chunk_ids]

    # ---- single-hop ----
    for cand_idx, (q, facts) in SINGLE_HOP.items():
        if cand_idx >= len(sh):
            continue
        sample = sh[cand_idx]
        e = sample["edge"]
        qid += 1
        chunks = list(e["text_unit_ids"])
        docs = list(e["raw_doc_ids"])
        rec = {
            "question_id": f"q{qid:03d}",
            "category": "single_hop",
            "question": q,
            "gold_chunk_ids": chunks,
            "gold_evidence_pages": [
                {"raw_doc_id": d, "pages": list(e["pages"])} for d in docs
            ],
            "gold_doc_titles": sorted({display_title(d) for d in docs}),
            "gold_facts": facts,
            "expected_behavior": "answer",
            "meta": {
                "candidate_idx": cand_idx,
                "edge_id": e["id"],
                "edge_type": e["type"],
                "src_label": sample["src"]["label"],
                "tgt_label": sample["tgt"]["label"],
            },
        }
        out_lines.append(rec)

    # ---- multi-hop ----
    for cand_idx, (q, facts) in MULTI_HOP.items():
        if cand_idx >= len(mh):
            continue
        sample = mh[cand_idx]
        e1 = sample["edge1"]
        e2 = sample["edge2"]
        qid += 1
        chunks = list(set(e1["text_unit_ids"]) | set(e2["text_unit_ids"]))
        docs = list(set(e1["raw_doc_ids"]) | set(e2["raw_doc_ids"]))
        # Gather pages per doc.
        page_map: dict[str, set] = {d: set() for d in docs}
        for d in e1["raw_doc_ids"]:
            page_map.setdefault(d, set()).update(e1["pages"])
        for d in e2["raw_doc_ids"]:
            page_map.setdefault(d, set()).update(e2["pages"])
        rec = {
            "question_id": f"q{qid:03d}",
            "category": "multi_hop",
            "question": q,
            "gold_chunk_ids": sorted(chunks),
            "gold_evidence_pages": [
                {"raw_doc_id": d, "pages": sorted(p)} for d, p in page_map.items()
            ],
            "gold_doc_titles": sorted({display_title(d) for d in docs}),
            "gold_facts": facts,
            "expected_behavior": "answer",
            "meta": {
                "candidate_idx": cand_idx,
                "A": sample["A"]["label"],
                "B": sample["B"]["label"],
                "C": sample["C"]["label"],
                "edge1_type": e1["type"],
                "edge2_type": e2["type"],
                "B_degree": sample["B_degree"],
            },
        }
        out_lines.append(rec)

    # ---- aggregation ----
    # We do NOT pre-list chunk_ids exhaustively for aggregation — the hub +
    # its typed neighbors span many chunks. Instead, we record the hub_id and
    # the neighbor type filter. Scoring will check that the answer covers >=N
    # of the listed gold_facts entries (semantic overlap).
    # We DO record all chunks attached to the hub node itself so retrieval
    # recall has a meaningful denominator.
    storage_chunks: dict[str, list[str]] = {}
    # Need access to actual node.text_unit_ids — load production.
    from src.graph.storage import GraphStorage
    gs = GraphStorage(ROOT / "data" / "production" / "graph.pkl")
    gs.load()
    nodes_by_id = {n.id: n for n in gs.nodes()}
    # Find hub neighbors and aggregate their chunks too (filtered by type).
    from src.graph.text_units import TextUnitStore
    tus = TextUnitStore(ROOT / "data" / "production" / "text_units.json")
    tus.load()

    for agg in AGGREGATION:
        hub = hubs[agg["hub_idx"]]
        hub_node = nodes_by_id[hub["node"]["id"]]
        chunks = set(hub_node.text_unit_ids or [])
        # Walk neighbors of the chosen type filter and union their chunks.
        from collections import defaultdict
        adj = defaultdict(list)
        for ed in gs.edges():
            adj[ed.source_id].append(ed.target_id)
            adj[ed.target_id].append(ed.source_id)
        for nb_id in set(adj[hub_node.id]):
            nb = nodes_by_id.get(nb_id)
            if nb is None or nb.merged_into is not None:
                continue
            if nb.type not in agg["neighbor_filter"]:
                continue
            chunks.update(nb.text_unit_ids or [])
        # Build evidence pages.
        page_map = defaultdict(set)
        for c in chunks:
            cd = tus.get(c)
            if cd is None:
                continue
            doc = cd.get("raw_doc_id", "")
            page_map[doc].add(cd.get("page_num"))

        qid += 1
        rec = {
            "question_id": f"q{qid:03d}",
            "category": "aggregation",
            "question": agg["question"],
            "gold_chunk_ids": sorted(chunks),
            "gold_evidence_pages": [
                {"raw_doc_id": d, "pages": sorted(p for p in pp if p is not None)}
                for d, pp in page_map.items()
            ],
            "gold_doc_titles": sorted({display_title(d) for d in page_map.keys()}),
            "gold_facts": agg["facts"],
            "expected_behavior": "answer",
            "meta": {
                "hub_label": hub["node"]["label"],
                "hub_type": hub["node"]["type"],
                "neighbor_type_filter": list(agg["neighbor_filter"]),
                "n_gold_chunks": len(chunks),
            },
        }
        out_lines.append(rec)

    # ---- oos ----
    for q in OOS:
        qid += 1
        rec = {
            "question_id": f"q{qid:03d}",
            "category": "oos",
            "question": q,
            "gold_chunk_ids": [],
            "gold_evidence_pages": [],
            "gold_doc_titles": [],
            "gold_facts": [],
            "expected_behavior": "refuse",
            "meta": {},
        }
        out_lines.append(rec)

    out_path = ROOT / "eval" / "testset" / "gold.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in out_lines:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    counts = {}
    for r in out_lines:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    print(f"wrote {out_path}")
    print(f"total: {len(out_lines)}")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
