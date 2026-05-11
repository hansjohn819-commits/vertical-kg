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
  domain: Person × Person
  semantics: two persons co-authored a publication together. Symmetric — direction has no semantic meaning, the inverse is the same predicate.
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
  domain: (Person | Organization) × Location
  semantics: entity performed research, fieldwork, or analysis focused on this location. Use only when the source explicitly ties the entity to studying that place — not for incidental mentions. Organizations include research institutes, government agencies running studies, NGOs publishing assessments.
  inverse: STUDIED_BY

## LOCATED_IN
  domain: (Organization | Company | Location) × Location
  semantics: physical or jurisdictional containment — entity exists within or is part of the target location. Distinct from HEADQUARTERED_IN which is specifically about an HQ.
  inverse: CONTAINS

## OPERATES_IN
  domain: (Company | Organization) × Location
  semantics: entity has business or operational activity in the location, even when not headquartered there. Multi-region companies / NGOs typically have many of these.
  inverse: HOSTS_OPERATIONS_OF

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
  domain: (Person | Organization) × Product
  semantics: the person or organization authored the product (paper, report, book, standard, dataset). Use for the author→work direction when there is a single author or when listing authors; use AUTHORED_WITH for co-authorship between two people. Organizations are included for corporate authorship (e.g., "FAO authored SOFIA 2024").
  inverse: AUTHORED

## PUBLISHED
  domain: (Organization | Company) × Product
  semantics: inverse of PUBLISHED_BY — organization issued or published the product. Same semantic content; the M1 LLM emits this direction on publisher-first sentences ("FAO published the report").
  inverse: PUBLISHED_BY

## EXPORTS
  domain: (Company | Organization | Location) × Product
  semantics: source entity exports the product to other regions. Locations cover country/region-level export statistics common in industry reports ("China exports sea kelp"). Companies and organizations cover firm-level exports.
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
