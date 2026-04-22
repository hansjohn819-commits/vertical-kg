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
  domain: Person × Company
  semantics: person is the chief executive officer of the company at time of assertion
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
  domain: Company × Product
  semantics: company manufactures or sells the product
  inverse: PRODUCED_BY

## HEADQUARTERED_IN
  domain: (Company | Organization) × Location
  semantics: primary administrative seat is at this location
  inverse: HOSTS_HQ_OF

## IN_INDUSTRY
  domain: (Company | Product) × Industry
  semantics: participates in or belongs to the industry sector
  inverse: INCLUDES

## REGULATES
  domain: Organization × (Company | Industry)
  semantics: has formal regulatory authority over the target
  inverse: REGULATED_BY

## OWNS
  domain: Person × Company
  semantics: controls majority ownership
  inverse: OWNED_BY

## RELATED_TO
  domain: any × any
  semantics: generic fallback edge for M4d link-form until a specific type is proposed
  inverse: RELATED_TO

# Global Conventions
- summary length: ≤300 tokens
- detail length: ≤8000 tokens
- merge candidate threshold: neighbor Jaccard > 0.3 OR summary embedding cos > 0.85
- type proposal threshold: ≥5 misclassified nodes in one pass
- prune thresholds: 3 consecutive passes marked suspicious → delete

# Evolution Log (append-only, LLM 维护)
- 2026-04-21: seed ontology (Phase 6 toy demo)
