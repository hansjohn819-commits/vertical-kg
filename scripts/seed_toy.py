"""Seed a toy knowledge graph for Phase 6 smoke tests.

~12 nodes with intentionally planted structure:
- Tesla Inc / Tesla Motors: two nodes for the same real-world entity (4b merge target)
- Musk shared across Tesla + SpaceX + Twitter (4d link-form candidate)
- A low-weight edge tagged suspicious prior to seeding (4a prune target)

Usage:
    python -m scripts.seed_toy
"""

from pathlib import Path

from src.graph.instance import GraphInstance
from src.graph.models import DirectProv, Edge, Node

RAW_DOC_ID = "toy-seed-v1"
EXTRACTION_RUN_ID = "toy-seed-run"


def _prov() -> DirectProv:
    return DirectProv(raw_doc_id=RAW_DOC_ID, extraction_run_id=EXTRACTION_RUN_ID)


def _node(type_: str, label: str, summary: str, detail: str = "", weight: float = 1.0) -> Node:
    return Node(
        type=type_, label=label, summary=summary, detail=detail,
        weight=weight, provenance=_prov(),
    )


def _edge(src_id: str, tgt_id: str, type_: str, weight: float = 1.0) -> Edge:
    return Edge(source_id=src_id, target_id=tgt_id, type=type_, weight=weight, provenance=_prov())


def seed(storage_path: str = "data/experiment") -> GraphInstance:
    gi = GraphInstance("toy-experiment", storage_path, "ontology.md")

    # Wipe any prior graph so the seed is deterministic.
    for n in list(gi.storage.nodes()):
        gi.storage.remove_node(n.id)

    # --- Entities ------------------------------------------------------
    tesla_inc = _node(
        "Company", "Tesla Inc",
        "American electric vehicle and clean energy company founded in 2003, headquartered in Austin since 2021.",
        "Tesla Inc is an American multinational automotive and clean energy company headquartered in Austin, Texas. "
        "Founded in 2003 by Martin Eberhard and Marc Tarpenning, with Elon Musk joining as chairman in 2004 and later "
        "becoming CEO. The company designs and manufactures electric vehicles, battery energy storage, and solar panels. "
        "Originally named Tesla Motors Inc, renamed to Tesla Inc in 2017.",
        weight=3.0,
    )
    tesla_motors = _node(
        "Company", "Tesla Motors",
        "Electric vehicle manufacturer founded 2003 by Eberhard and Tarpenning; former legal name of what is now Tesla Inc.",
        "Tesla Motors was the original legal name of the American electric vehicle company founded July 1, 2003. "
        "Co-founded by Martin Eberhard, Marc Tarpenning, JB Straubel, Ian Wright, and Elon Musk. "
        "The company was renamed Tesla Inc in February 2017 to reflect its expansion beyond automotive into energy. "
        "All products, facilities, and personnel were continuous across the rename — same company, new legal name.",
        weight=2.5,
    )
    musk = _node(
        "Person", "Elon Musk",
        "Entrepreneur; CEO of Tesla since 2008, founder of SpaceX, owner of X (formerly Twitter).",
        "Elon Reeve Musk (born 1971, Pretoria) is an entrepreneur and business magnate. "
        "Co-founder of PayPal, founder and CEO of SpaceX, CEO and product architect of Tesla Inc since 2008, "
        "acquired Twitter in 2022 and renamed it X.",
        weight=3.0,
    )
    straubel = _node(
        "Person", "JB Straubel",
        "Co-founder and former CTO of Tesla; founder of Redwood Materials.",
        "Jeffrey Brian Straubel is an American engineer. Co-founded Tesla Motors in 2003 and served as CTO until 2019. "
        "Founded battery-recycling firm Redwood Materials in 2017.",
        weight=1.5,
    )
    spacex = _node(
        "Company", "SpaceX",
        "American spacecraft manufacturer and space transportation company founded by Musk in 2002.",
        "",
        weight=2.0,
    )
    x_corp = _node(
        "Company", "X",
        "Social media platform, renamed from Twitter after Musk's 2022 acquisition.",
        "",
        weight=1.5,
    )
    model_s = _node(
        "Product", "Model S",
        "Tesla's flagship full-size electric sedan, launched 2012.",
        "",
        weight=1.2,
    )
    model_3 = _node(
        "Product", "Model 3",
        "Tesla's mass-market compact electric sedan, launched 2017.",
        "",
        weight=1.2,
    )
    palo_alto = _node(
        "Location", "Palo Alto",
        "City in California's Silicon Valley; Tesla's former headquarters location.",
        "",
    )
    austin = _node(
        "Location", "Austin",
        "Capital city of Texas; Tesla's headquarters since 2021.",
        "",
    )
    nhtsa = _node(
        "Organization", "NHTSA",
        "U.S. National Highway Traffic Safety Administration; regulates automotive safety in the United States.",
        "",
    )
    ev_industry = _node(
        "Industry", "Electric Vehicles",
        "Automotive industry segment covering battery-electric passenger and commercial vehicles.",
        "",
    )

    for n in [tesla_inc, tesla_motors, musk, straubel, spacex, x_corp,
              model_s, model_3, palo_alto, austin, nhtsa, ev_industry]:
        gi.storage.add_node(n)

    # --- Edges ---------------------------------------------------------
    edges = [
        _edge(musk.id, tesla_inc.id, "CEO_OF", weight=2.0),
        _edge(tesla_motors.id, model_s.id, "PRODUCES", weight=1.5),
        _edge(tesla_inc.id, model_3.id, "PRODUCES", weight=1.5),
        _edge(musk.id, spacex.id, "FOUNDED", weight=2.0),
        _edge(musk.id, x_corp.id, "OWNS", weight=1.5),
        _edge(tesla_inc.id, austin.id, "HEADQUARTERED_IN", weight=1.5),
        _edge(tesla_motors.id, palo_alto.id, "HEADQUARTERED_IN", weight=1.3),
        _edge(straubel.id, tesla_motors.id, "CO_FOUNDED", weight=1.5),
        _edge(model_s.id, ev_industry.id, "IN_INDUSTRY", weight=1.0),
        _edge(model_3.id, ev_industry.id, "IN_INDUSTRY", weight=1.0),
        _edge(nhtsa.id, tesla_inc.id, "REGULATES", weight=1.2),
    ]

    # A low-weight edge pre-marked suspicious — 4a should finish it off.
    stale = _edge(musk.id, nhtsa.id, "RELATED_TO", weight=0.05)
    stale.suspicious = True
    stale.suspicious_pass_count = 2
    edges.append(stale)

    for e in edges:
        gi.storage.add_edge(e)

    gi.save()
    stats = gi.stats()
    print(f"Seeded {stats['node_count']} nodes, {stats['edge_count']} edges into {storage_path}/graph.pkl")
    print(f"  node_types: {stats['node_types']}")
    print(f"  edge_types: {stats['edge_types']}")
    return gi


if __name__ == "__main__":
    seed()
