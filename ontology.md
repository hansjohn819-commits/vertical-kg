# Entity Types

## Company
  required: [label]
  optional: [founded_year, hq]
  when to use: commercial corporations, subsidiaries, joint ventures
  when NOT to use: public-sector regulators (use Organization), non-corporate groups
  merge rule: label canonical form (lowercase, strip "Inc/Corp/Motors/LLC") matches AND ≥1 shared neighbor

## Person
  required: [label]
  optional: [role]
  when to use: individual humans identified by name
  when NOT to use: fictional characters, generic roles ("CEO" without a name)
  merge rule: label full-name match (case-insensitive); same role at same company is strong signal

## Product
  required: [label]
  optional: [launch_year]
  when to use: named commercial products or model lines
  when NOT to use: product categories ("sedan") — use Industry or a concept type
  merge rule: exact label match; do NOT merge different trims of the same line

## Location
  required: [label]
  optional: [country]
  when to use: named geographic places (city, region, country)
  when NOT to use: building-level addresses
  merge rule: label match after normalization; disambiguate by country when label collides

## Organization
  required: [label]
  optional: [kind]
  when to use: government agencies, NGOs, regulators, non-corporate institutional bodies
  when NOT to use: for-profit corporations (use Company)
  merge rule: canonical acronym match OR full-name match

## Industry
  required: [label]
  when to use: named industry / sector groupings
  merge rule: label match after normalization

# Relation Types

## CEO_OF
  domain: Person × (Company | Organization)
  semantics: person is the chief executive officer of the entity at time of assertion. Includes non-profit / NGO presidents and executive directors that the source describes as "CEO".
  inverse: HAS_CEO

## FOUNDED
  domain: Person × (Company | Organization)
  semantics: person was a founding member of the entity
  inverse: FOUNDED_BY

## CO_FOUNDED
  domain: Person × (Company | Organization)
  semantics: person was one of multiple founders
  inverse: CO_FOUNDED_BY

## PRODUCES
  domain: (Company | Organization) × Product
  semantics: entity manufactures, sells, or publishes the product. Organizations are included to cover regulators / standards bodies that produce reference documents (e.g., ASC-MSC Seaweed Standard).
  inverse: PRODUCED_BY

## HEADQUARTERED_IN
  domain: (Company | Organization) × Location
  semantics: primary administrative seat is at this location
  inverse: HOSTS_HQ_OF

## IN_INDUSTRY
  domain: (Company | Product | Location) × Industry
  semantics: participates in or belongs to the industry sector. Locations are included to cover regional industry membership ("Maine is in the seaweed industry").
  inverse: INCLUDES

## REGULATES
  domain: Organization × (Company | Industry)
  semantics: has formal regulatory authority over the target
  inverse: REGULATED_BY

## OWNS
  domain: Person × Company
  semantics: controls majority ownership
  inverse: OWNED_BY

## AUTHORED_WITH
  domain: (Person | Organization) × (Person | Organization)
  semantics: two parties co-authored a publication together. Symmetric — direction has no semantic meaning, the inverse is the same predicate. Person × Person covers academic co-authorship; Organization × Organization covers joint institutional reports ("FAO, IFAD and UNICEF co-authored the State of Food Security report"); mixed Person × Organization is also accepted when an individual co-signs with a body.
  inverse: AUTHORED_WITH

## AFFILIATED_WITH
  domain: (Person | Organization | Company) × (Organization | Company)
  semantics: source entity has a current or recent institutional affiliation with the target. Covers both individual affiliations (researcher at a university) and inter-org affiliations (a local salmon club is an "affiliate" of a national federation).
  inverse: HAS_AFFILIATE

## PUBLISHED_BY
  domain: Product × (Organization | Company)
  semantics: product (paper, report, standard, dataset, model) was issued or published by the organization. Used for grey literature, journal publishers, standards bodies, etc.
  inverse: PUBLISHED

## STUDIED_LOCATION
  domain: (Person | Organization | Product | Industry) × (Location | Industry)
  semantics: entity performed research, fieldwork, or analysis focused on this region or industry sector. Use only when the source explicitly ties the entity to studying the target — not for incidental mentions. Organizations include research institutes, government agencies running studies, NGOs publishing assessments. Products are included for reports, studies, and datasets whose subject scope is a named location or industry ("State of the Kelp Industry studies the Gulf of Maine", "SOFIA 2024 studies fisheries"). Industry as src covers sector-level research bodies ("seaweed production research focused on North-West Europe").
  inverse: STUDIED_BY

## LOCATED_IN
  domain: (Person | Organization | Company | Location | Industry) × Location
  semantics: physical or jurisdictional containment — entity exists within or is part of the target location. Distinct from HEADQUARTERED_IN which is specifically about an HQ. Person covers residence / current base ("Kelly Hinkle is located in Maine"); Industry covers regional industry sectors ("Maine seaweed sector LOCATED_IN Maine").
  inverse: CONTAINS

## OPERATES_IN
  domain: (Company | Organization) × Location
  semantics: entity has business or operational activity in the location, even when not headquartered there. Multi-region companies / NGOs typically have many of these.
  inverse: HOSTS_OPERATIONS_OF

## HOSTS_OPERATIONS_OF
  domain: Location × (Company | Organization)
  semantics: inverse of OPERATES_IN — the location hosts the operations of the source entity. The M1 LLM emits this direction on location-first sentences ("New Brunswick hosts Cooke Aquaculture's operations"). Same semantic content as OPERATES_IN, just the other arrow.
  inverse: OPERATES_IN

## AUDITED_BY
  domain: (Organization | Company) × (Organization | Company)
  semantics: source entity has had its financial statements or operations audited by the target (typically an external accounting firm or oversight body). Common in NGO annual reports.
  inverse: AUDITS

## AUDITS
  domain: (Organization | Company) × (Organization | Company)
  semantics: inverse of AUDITED_BY — source entity audits the target.
  inverse: AUDITED_BY

## PRODUCED_BY
  domain: Product × (Company | Organization)
  semantics: inverse of PRODUCES — product is manufactured, published, or issued by the entity. Same semantic content as PRODUCES, just the other direction; the M1 LLM often emits this direction directly on product-first sentences.
  inverse: PRODUCES

## PRODUCED_IN
  domain: Product × Location
  semantics: product is produced, harvested, cultivated, or otherwise originates in the location. Use for "Saccharina japonica is cultivated in Europe", "kelp produced in Maine". Distinct from LOCATED_IN (which is for organizations and locations) and from STUDIED_LOCATION (which is about research focus).
  inverse: PRODUCES_LOCATION

## AUTHORED_BY
  domain: (Person | Organization | Company) × Product
  semantics: the person, organization, or company authored the product (paper, report, book, standard, dataset). Use for the author→work direction when there is a single author or when listing authors; use AUTHORED_WITH for co-authorship between two parties. Organizations are included for corporate authorship (e.g., "FAO authored SOFIA 2024"); Companies are included for consulting firms / industry analysts ("Maritime Blue & Ocean Strategies authored the Washington Seaweed Aquaculture Economic Potential Analysis").
  inverse: AUTHORED

## PUBLISHED
  domain: (Organization | Company) × Product
  semantics: inverse of PUBLISHED_BY — organization issued or published the product. Same semantic content; the M1 LLM emits this direction on publisher-first sentences ("FAO published the report").
  inverse: PUBLISHED_BY

## EXPORTS
  domain: (Company | Organization | Location) × (Product | Location)
  semantics: source entity exports goods to the target. Product target covers the commodity-level pattern ("China exports sea kelp"); Location target covers destination-pair trade statistics common in trade reports ("Ecuador exports to the United States"). Companies and organizations cover firm-level exports.
  inverse: IMPORTED_FROM

## INCLUDES
  domain: Industry × (Company | Product | Location)
  semantics: inverse of IN_INDUSTRY — industry sector includes the member entity. M1 emits this direction on industry-first sentences ("the seaweed industry includes Atlantic Sea Farms").
  inverse: IN_INDUSTRY

## PART_OF
  domain: any × any
  semantics: generic structural containment — source entity is a constituent or sub-unit of the target. Polymorphic by design: covers sub-region → region (Location × Location), department → parent organization (Organization × Organization), sub-project → initiative (Product × Product), etc. Prefer a more specific relation when one exists (LOCATED_IN for geographic containment, AFFILIATED_WITH for institutional membership, IN_INDUSTRY for industry sector).
  inverse: HAS_PART

## FUNDED_BY
  domain: (Person | Organization | Company | Product) × (Organization | Company)
  semantics: source entity received funding, grants, or financial support from the target. Products are included because grants are often described as funding a specific project, paper, or program ("research funded by NSF"). Common in NGO annual reports and academic acknowledgments.
  inverse: FUNDED

## FUNDED
  domain: (Organization | Company) × (Person | Organization | Company | Product | Industry)
  semantics: inverse of FUNDED_BY — source entity provides funding, grants, or financial support to the target. Same semantic content as FUNDED_BY, just the other direction; the M1 LLM emits this direction on funder-first sentences ("Green Climate Fund funded the EAF-Nansen Programme").
  inverse: FUNDED_BY

## IMPORTED_FROM
  domain: (Company | Organization | Location) × (Company | Organization | Location | Product)
  semantics: source entity imports goods or products from the target. Locations cover country/region-level import statistics common in trade reports ("the European Union imported from Norway"); Product target covers commodity-level pattern ("Africa imports mackerels"), symmetric with EXPORTS having Product in its target. Inverse of EXPORTS at the relation level; both directions are emitted by the M1 LLM depending on sentence form.
  inverse: EXPORTS

## PARTNERED_WITH
  domain: (Person | Organization | Company | Product) × (Organization | Company | Product)
  semantics: source entity has a named partnership, collaboration agreement, or co-implementation arrangement with the target. Distinct from AFFILIATED_WITH (institutional affiliation / membership) and CONTRACTED_WITH (formal contractual procurement). Products are valid on either side for programs / initiatives described as "partnered with" ("FISH4ACP partnered with Chinhoyi University of Technology", "International Maritime Organization partnered with GloLitter Partnerships Project"). Symmetric in intent — direction reflects the sentence form, not a semantic asymmetry.
  inverse: PARTNERED_WITH

## CONTAINS
  domain: (Location | Organization | Product | Industry) × (Location | Organization | Product | Industry | Company)
  semantics: source entity structurally contains the target. Polymorphic: covers geographic containment (Location × Location), organizational sub-units (Organization × Organization), report sections / annexes (Product × Product), and industry coverage of constituent entities. Prefer a more specific relation when one exists (LOCATED_IN's inverse for geographic, AFFILIATED_WITH for institutional membership, INCLUDES for industry sector, PART_OF's inverse for generic containment).
  inverse: LOCATED_IN

## PRODUCES_LOCATION
  domain: Location × Product
  semantics: inverse of PRODUCED_IN — location produces, harvests, cultivates, or otherwise originates the product. Same semantic content as PRODUCED_IN, just the other direction; the M1 LLM emits this direction on location-first sentences ("Maine produces kelp", "China exports sea kelp").
  inverse: PRODUCED_IN

## RELATED_TO
  domain: any × any
  semantics: generic fallback edge for M4d link-form until a specific type is proposed
  inverse: RELATED_TO

# Aliases (relation type renames; auto-applied at ingest time)
# Format: ALIAS → CANONICAL. The M1 ingest pipeline rewrites every
# proposed type matching ALIAS to its CANONICAL form before domain
# validation, so model typos and surface variants don't fragment the
# edge-type vocabulary. Add new lines here as typos surface in the
# `kind=ontology_proposal` log stream.
- AFFULIATED_WITH → AFFILIATED_WITH
- AFFILIATIED_WITH → AFFILIATED_WITH
- AUTHORED → AUTHORED_BY
- IMPORT_FROM → IMPORTED_FROM
- PARTNERS_WITH → PARTNERED_WITH
- OPERATESS_IN → OPERATES_IN

# Global Conventions
- summary length: ≤300 tokens
- detail length: ≤8000 tokens
- merge candidate threshold: neighbor Jaccard > 0.3 OR summary embedding cos > 0.85
- type proposal threshold: ≥5 misclassified nodes in one pass
- prune thresholds: 3 consecutive passes marked suspicious → delete

# Evolution Log (append-only, LLM 维护)
- 2026-04-21: seed ontology (Phase 6 toy demo)
- 2026-05-09: first round of evidence-driven evolution after the §16.17 ingest pipeline started logging `kind=ontology_proposal` events. Triggered by the Kelponomics_The_State_of_North_American_Kelp_Maric.pdf 34-page run (run_id m1-b1b6c0b9): 91 relation_type_proposed events spanning 23 unique types, 13 domain_mismatch events.
  - **Extended `PRODUCES` domain** from `Company × Product` to `(Company | Organization) × Product`. Trigger: 8 domain_mismatch events for `Organization → Product` pairs like `Aquaculture Stewardship Council → ASC-MSC Seaweed Standard` and `Marine Stewardship Council → ASC-MSC Seaweed (Algae) Standard`. Standards bodies do produce things, even if non-commercial.
  - **Added `AUTHORED_WITH`** (Person × Person, symmetric). 32 proposals in one run; standard academic co-authorship relation that came up immediately on a research-heavy source.
  - **Added `AFFILIATED_WITH`** (Person × (Organization | Company)). 12 proposals; needed for "researcher at Simon Fraser University" type claims that we don't want to coerce into `OWNS` or `HEADQUARTERED_IN`.
  - **Added `PUBLISHED_BY`** (Product × (Organization | Company)). 9 proposals; covers academic journals, standards bodies, and grey-literature publishers. Treats publications as Products until a dedicated `Publication` entity type warrants its own block.
  - **Added `STUDIED_LOCATION`** (Person × Location). 9 proposals; specific to research / fieldwork claims. Distinct from incidental mentions or affiliation.
  - **Added `LOCATED_IN`** ((Organization | Company | Location) × Location). 3 proposals plus broader semantic need; complements `HEADQUARTERED_IN` for non-HQ containment (subsidiary office, regional branch, sub-region).
  - **Added `OPERATES_IN`** ((Company | Organization) × Location). 3 proposals; captures business activity in a location without implying HQ. Phyconomy industry distribution analysis used this naturally.
  - Deferred for further evidence: `PART_OF_CLUSTER` (6 proposals but document-specific to PESTLE clustering), `PARTNERS_WITH` / `MEMBERSHIP_OF` / `PREPARED_FOR` (≤2 proposals each — wait for cross-document confirmation before promoting). 1 `entity_type_proposed: Concept` also deferred.
- 2026-05-09 (later same day): second round, triggered by 2025_ImpactReport_web_pages.pdf (m1-a249deed) + ASF-Impact-Report-2024_WEB.pdf (m1-238c0641) runs. ASF is a non-profit annual report — Organization-heavy (vs Kelponomics's Company/academic mix), which surfaced two large domain gaps the first round didn't see.
  - **Extended `CEO_OF` domain** from `Person × Company` to `Person × (Company | Organization)`. Trigger: 23 domain_mismatch events for `Person → Organization` pairs like `John Thompson → ASF (CANADA)`. Non-profits use the title "CEO" too.
  - **Extended `AFFILIATED_WITH` domain** from `Person × (Organization | Company)` to `(Person | Organization | Company) × (Organization | Company)`. Trigger: 94 domain_mismatch events, dominantly `Org → Org` like `Restigouche Salmon Club → ASF Affiliates`. NGOs federate; affiliations aren't only person-level.
  - **Extended `STUDIED_LOCATION` src** from `Person` to `(Person | Organization)`. 10 mismatches; institutional research on a region is as common as individual fieldwork in this corpus.
  - **Added `AUDITED_BY`** ((Organization | Company) × (Organization | Company)) and inverse **`AUDITS`**. Trigger: 6 + 2 proposals from ASF annual report (external accounting audit relationships are standard NGO disclosure content).
  - **Added `# Aliases` section** with two model-typo aliases observed this round: `AFFULIATED_WITH → AFFILIATED_WITH`, `AFFILIATIED_WITH → AFFILIATED_WITH`. Both are clear spelling errors of an already-registered type. The aliases get auto-applied at M1 ingest time so the typos don't fragment the edge-type vocabulary.
  - Deferred for further evidence: `OPERATES_WITH` / `WORKING_WITH` / `COLLABORATES_WITH` (1-2 each — semantic overlap with `PARTNERS_WITH`, wait to see which surface form dominates), `PART_OF_SERIES` (2, document-series specific), `FUNDED_BY` (2, was also seen in Kelponomics — promote next round if it appears again).
- 2026-05-11: third round, triggered by completing the FAO SOFIA 2024 264-page ingest (4 super-chunks, run_ids m1-c55263b2 / m1-c7cdad3a / m1-2fbbed7a / m1-63307ad7) plus accumulated signals from the previous four documents. Aggregated proposals across the corpus produced 89 distinct relation types and 830 domain-mismatch events. Eight new relations and one alias added; the IN_INDUSTRY domain was extended.
  - **Added `PRODUCED_BY`** (Product × (Company | Organization)). 38 proposals — the M1 model frequently emits the product→producer direction on product-first sentences, mirroring PRODUCES. 32 such edges already lived in the graph as free-form types.
  - **Added `PRODUCED_IN`** (Product × Location). Synthesized to absorb a large class of cross-domain mismatches: 79 `OPERATES_IN Product→Location` + 55 `LOCATED_IN Product→Location` + 35 `STUDIED_LOCATION Product→Location` + 34 `PRODUCES Location→Product` all describe the same semantic ("this product is cultivated/harvested/produced in this place") but were getting downgraded to RELATED_TO under three different relation labels. PRODUCED_IN gives the M1 model a target type that matches the natural sentence form.
  - **Added `AUTHORED_BY`** ((Person | Organization) × Product). 23 direct proposals + 31 mismatches under `AUTHORED_WITH Person→Product` = 54 signals. AUTHORED_WITH is co-authorship between two people; AUTHORED_BY is author → work. Both forms are needed and overlap cleanly.
  - **Added `PUBLISHED`** ((Organization | Company) × Product). 116 domain mismatches on `PUBLISHED_BY Organization→Product` — by far the highest-volume mismatch in the corpus. The model emits this direction on publisher-first sentences ("FAO published the report"). Same semantic as PUBLISHED_BY, just the other arrow.
  - **Added `EXPORTS`** ((Company | Organization | Location) × Product). 15 proposals + 14 existing edges. Country/region-level export statistics are common in industry reports.
  - **Added `INCLUDES`** (Industry × (Company | Product | Location)). 25 proposals + 24 existing edges. Inverse of IN_INDUSTRY; emitted on industry-first sentences.
  - **Added `PART_OF`** (any × any). 14 proposals + 13 existing edges. Polymorphic structural containment with an explicit note in the semantics line to prefer specific relations (LOCATED_IN / AFFILIATED_WITH / IN_INDUSTRY) when applicable.
  - **Added `FUNDED_BY`** ((Person | Organization | Company | Product) × (Organization | Company)). 9 proposals this round — last round deferred at 2 with the note "promote next round if it appears again". Cross-document and now well past the ≥5 threshold.
  - **Extended `IN_INDUSTRY` src** from `(Company | Product)` to `(Company | Product | Location)`. Trigger: 14 `Location → Industry` mismatches (regional industry membership).
  - **Added alias** `AUTHORED → AUTHORED_BY`. 6 proposals of bare `AUTHORED` — same semantic as the just-added AUTHORED_BY, treat as a surface variant.
  - Deferred for further evidence: `PART_OF_CLUSTER` (12 cross-document but still PESTLE-specific), `USED_IN_PRODUCTION_OF` / `CONTRIBUTED_DATA_TO` / `PUBLISHED_IN` (5-6 each, FAO SOFIA-specific patterns — wait for non-FAO confirmation), `MEMBERSHIP_OF` / `CUSTODIAN_OF` / `PART_OF_SERIES` / `PARTNERS_WITH` (3-4 each, still below threshold), entity types `Concept` (50) and `Entity` (17) (both too generic — wait for a concrete structural need).
  - Audit note: 83 `AFFILIATED_WITH Organization→Organization` and 23 `CEO_OF Person→Organization` mismatches remain in this log window, but both domains were already extended in round 2; these signals span the round-1/round-2 boundary and should disappear after `scripts/normalize_edges.py` is re-run against the current ontology.
- 2026-05-11 (later same day): fourth round, triggered by ingest of five additional documents post round 3 (Maine-Seaweed-Benchmarking, Washington-Seaweed-Aquaculture-Economic-Potential, State-of-the-Kelp-Industry-Report_Feb-2026 [80p + tail], TEA, doc.pdf) plus a retroactive `normalize_edges` and an M4d link_form pass. Aggregated 239 relation_type_proposed across 121 distinct types and 915 domain_mismatch events. Five new relations + two aliases added; STUDIED_LOCATION src extended.
  - **Added `FUNDED`** ((Organization | Company) × (Person | Organization | Company | Product | Industry)). 9 proposals across 3 documents. Inverse of FUNDED_BY (added round 3); inverse name was already declared but never registered as its own type. Same pattern as PUBLISHED / PUBLISHED_BY and PRODUCES / PRODUCED_BY.
  - **Added `IMPORTED_FROM`** ((Company | Organization | Location) × (Company | Organization | Location)). 5 direct proposals + 5 `IMPORT_FROM` typo variants. Inverse of EXPORTS was declared but unregistered. Country-level import statistics are a standard pattern in trade reports.
  - **Added `PARTNERED_WITH`** ((Person | Organization | Company | Product) × (Organization | Company)). 6 `PARTNERED_WITH` + 1 `PARTNERS_WITH` across 3 documents. Deferred twice in rounds 1 and 2 ("≤2 proposals each — wait for cross-document confirmation"); now well past threshold. Distinct from AFFILIATED_WITH (membership/affiliation) and CONTRACTED_WITH (procurement contract).
  - **Added `CONTAINS`** ((Location | Organization | Product | Industry) × (Location | Organization | Product | Industry | Company)). 10 proposals across 3 documents. Was declared as `LOCATED_IN.inverse: CONTAINS` (round 1) but unregistered; also picks up structural containment beyond geography (report sections, organizational sub-units).
  - **Added `PRODUCES_LOCATION`** (Location × Product). 3 direct proposals + 92 `PRODUCED_IN Location→Product` domain_mismatch events. Inverse of PRODUCED_IN (added round 3); inverse name was already declared. The M1 model emits this direction on location-first sentences, identical pattern to PUBLISHED / PUBLISHED_BY.
  - **Extended `STUDIED_LOCATION` src** from `(Person | Organization)` to `(Person | Organization | Product)`. Trigger: 91 `Product → Location` domain_mismatch events (reports / studies whose subject scope is a named region, e.g., "State of the Kelp Industry → Gulf of Maine"). Distinct from PRODUCED_IN — the product is the research instrument, not the harvested good.
  - **Added aliases** `IMPORT_FROM → IMPORTED_FROM` (5 occurrences, surface variant of newly registered IMPORTED_FROM) and `PARTNERS_WITH → PARTNERED_WITH` (1 occurrence; will catch the recurring tense variant before it fragments the vocabulary).
  - Deferred for further evidence: `USED_IN_PRODUCTION_OF` / `CONTRIBUTED_DATA_TO` (5 each, FAO SOFIA single-document — deferred third round, still no non-FAO confirmation), `CONTRACTED_WITH` (5, single-document Washington TEA), `COMPARED_TO` (8, dominantly `Concept→Concept` — gated on Concept entity decision), `PART_OF_CLUSTER` (6, PESTLE-specific, deferred third consecutive round), `CONTAINS` polymorphism into Industry domain (kept narrow to evidence-supported types this round), `entity_type_proposed: Concept` (9 this round, cumulative ~59 — exceeds threshold but introducing the type would broadly disrupt existing domains; needs a coordinated Concept + Concept-domain relations promotion, scheduled for round 5).
  - Audit note: the large mismatch pile on `OPERATES_IN Product→Location` (62) / `LOCATED_IN Product→Location` (41) / `AFFILIATED_WITH Organization→Location` (49) / `IN_INDUSTRY Industry→Location` (32) is `normalize_edges.py` residue, not an ontology gap — round 3 already added PRODUCED_IN / INCLUDES / extended IN_INDUSTRY src to cover these. Will dissolve after the next normalize_edges run against the current ontology.
- 2026-05-17: fifth round, triggered by ingest of three new documents post round 4 (occupational_health_and_safety_in_aquaculture, Facilitating-development-of-the-seaweed-cultivation-sector-in-Scotland-Feb-2022, Global-status-of-seaweed-production-trade-and-utilization-Junning-Cai-FAO). Aggregated proposals across the three runs: 0 new entity types (the existing six cover the new content), 23 relation_type_proposed across 11 distinct types (mostly single-document, deferred), and 343 domain_mismatch events concentrated on a handful of patterns. Eight domain extensions + one new relation + one alias added.
  - **Extended `AUTHORED_BY` src** from `(Person | Organization)` to `(Person | Organization | Company)`. Trigger: 21 `Company → Product` mismatches across 3 documents — consulting firms / industry analysts authoring corporate reports ("Maritime Blue & Ocean Strategies, Inc. → Washington Seaweed Aquaculture Economic Potential Analysis"). Standard pattern in grey literature that the original Person/Org-only domain was rejecting.
  - **Extended `IMPORTED_FROM` tgt** to include `Product`. Trigger: 14 `Location → Product` mismatches across 3 documents — commodity-level import flows ("Africa imports mackerels"). Symmetric with EXPORTS already having Product in its target.
  - **Extended `LOCATED_IN` src** from `(Organization | Company | Location)` to `(Person | Organization | Company | Location | Industry)`. Trigger: 12 `Industry → Location` mismatches across 3 documents ("Maine seaweed sector LOCATED_IN Maine") + 9 `Person → Location` mismatches across 3 documents ("Kelly Hinkle LOCATED_IN Maine"). Two extensions in one bullet.
  - **Extended `STUDIED_LOCATION`** — src adds `Industry`, tgt adds `Industry`. Trigger: 9 `Industry → Location` mismatches across 3 documents ("seaweed production studied in North West Europe") + 8 `Product → Industry` mismatches across 3 documents ("SOFIA 2024 studies fisheries"). Industry as both ends of a "studied" relation is natural for sector-level research bodies and for cross-sector studies.
  - **Extended `AUTHORED_WITH` domain** from `Person × Person` to `(Person | Organization) × (Person | Organization)`. Trigger: 6 `Organization → Organization` mismatches across 2 documents ("FAO authored_with IFAD / UNICEF"). Joint institutional reports are common in the FAO corpus.
  - **Added `HOSTS_OPERATIONS_OF`** (Location × (Company | Organization)). Trigger: 7 `OPERATES_IN: Location → Company` mismatches across 3 documents ("New Brunswick → Cooke Aquaculture"). Was declared as `OPERATES_IN.inverse: HOSTS_OPERATIONS_OF` (round 1) but unregistered. Same pattern as PUBLISHED / PUBLISHED_BY and PRODUCES_LOCATION / PRODUCED_IN — the M1 model emits this direction on location-first sentences.
  - **Extended `EXPORTS` tgt** from `Product` to `(Product | Location)`. Trigger: 4 `Location → Location` mismatches across 3 documents ("Ecuador → United States") + 4 `Company → Location` mismatches in Scotland ("SFO → Europe / Far East"). Destination-pair trade statistics are a standard pattern in trade reports.
  - **Extended `PARTNERED_WITH` tgt** from `(Organization | Company)` to `(Organization | Company | Product)`. Trigger: 11 `Organization → Product` mismatches across 3 documents ("International Maritime Organization → GloLitter Partnerships Project"). Programs / initiatives are valid partnership targets, symmetric with the src side which already allowed Product.
  - **Added alias** `OPERATESS_IN → OPERATES_IN` (1 typo from the OHS doc; cheap to register before it fragments).
  - Deferred for further evidence: `SUPPLIES` (Org → Co, 8 in Scotland — single-document, wait for cross-doc), `ASSISTED_WITH` (4, Scotland), `OWNED_BY` (3, Scotland — distinct from existing OWNS inverse name; defer), `CONTRACTED_WITH` (1, Scotland — third consecutive defer), `LEAD_PROJECT` / `SIBLING_OF` / `ADMINISTRATES` / `ADMINISTER_RULES_OF` / lowercase `part_of` / meta `ALIAS` (all 1-event noise).
  - Not adopted as ontology changes (semantic / typing errors, not gaps): `PRODUCED_IN: Location → Product` (43, already covered by PRODUCES_LOCATION — normalize_edges residue from round 4); `LOCATED_IN: Product → Location` (17, Walton's Mill Dam mistyped as Product); `PARTNERED_WITH: Organization → Location` (17, semantically OPERATES_IN); `AFFILIATED_WITH: Organization → Location` (9, semantically OPERATES_IN); `STUDIED_LOCATION: Location → Product` (5, direction error); `LOCATED_IN: Organization → Organization` (5, semantically AFFILIATED_WITH); `IN_INDUSTRY: Industry → Location` (4, direction error); `FUNDED_BY: Organization → Product` (3, direction error — use FUNDED); `INCLUDES: Product → Industry` (5, type-coercion error on LFCS).
  - Single-document type-coercion noise (Global-FAO labels "Seaweeds" as Industry instead of Product, producing 19 `EXPORTS: Location → Industry` and 18 `PRODUCES_LOCATION: Location → Industry` mismatches): these will dissolve when downstream M4b merges fold "Seaweeds (Industry)" into the existing "Seaweed" / "Seaweeds (Product)" node, or via a manual reclassify pass.
  - Audit note: the round-5 changes should flush most of this run's RELATED_TO residue. Run `scripts/normalize_edges.py` + `scripts/dedupe_edges.py` immediately after this commit to retroactively rewrite the edges that the three new runs downgraded under the previous ontology.
