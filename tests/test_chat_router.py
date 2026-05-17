"""GraphAgent chat router tests (post-2026-05-15 refactor).

The agent no longer uses LLM-driven tool selection. Chat input is routed by
hardcoded string matching:

    /ingest <filename>  → ingest the file
    /ingest              → list available files in data/raw/
    /sleep / /sleep pass → trigger sleep pass
    anything else        → instance.qa()

These tests cover the parsing + dispatch logic without requiring a running
LLM (`instance.qa` is monkeypatched). End-to-end Q&A is exercised by the
production smoke probe (see scripts/) which does need the LLM up.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

from src.modules.m2_qa_agent import GraphAgent, _INGEST_RE, _SLEEP_RE


# ---- regex sanity ----------------------------------------------------------

def test_sleep_regex_matches_variants():
    assert _SLEEP_RE.match("/sleep")
    assert _SLEEP_RE.match("/sleep pass")
    assert _SLEEP_RE.match("  /sleep  ")
    assert _SLEEP_RE.match("/SLEEP")
    assert _SLEEP_RE.match("/Sleep Pass")

def test_sleep_regex_rejects_misc():
    assert not _SLEEP_RE.match("sleep")           # missing slash
    assert not _SLEEP_RE.match("/sleeping")        # extra letters
    assert not _SLEEP_RE.match("Tell me /sleep")   # not at start
    assert not _SLEEP_RE.match("/sleep me")        # extra arg


def test_ingest_regex_matches_with_filename():
    m = _INGEST_RE.match("/ingest report.pdf")
    assert m and m.group(1) == "report.pdf"

    m = _INGEST_RE.match("/ingest a file with spaces.pdf")
    assert m and m.group(1) == "a file with spaces.pdf"

    m = _INGEST_RE.match("/INGEST   X.PDF  ")
    assert m and (m.group(1) or "").strip() == "X.PDF"


def test_ingest_regex_no_filename():
    m = _INGEST_RE.match("/ingest")
    assert m and (m.group(1) or "") == ""

    m = _INGEST_RE.match("  /ingest   ")
    assert m and (m.group(1) or "").strip() == ""


# ---- router dispatch -------------------------------------------------------

def _make_agent_with_mock_instance():
    inst = MagicMock()
    inst.sleep_pass_running = False
    inst.qa.return_value = "ANSWER_FROM_QA"
    inst.sleep_pass.return_value = {
        "pass_id": "test-pass-1",
        "stats": {"merge_total": 3, "edges_pruned_total": 5,
                  "nodes_pruned_total": 1, "new_links_total": 2,
                  "reinforced_total": 12},
    }
    agent = GraphAgent(inst)
    # Patch ingest so we don't touch the filesystem
    agent._ingest_file = MagicMock(return_value={
        "status": "ok", "filename": "foo.pdf",
        "pages_processed": 3, "nodes_added": 7, "edges_added": 4,
    })
    agent._list_raw_files = MagicMock(return_value={
        "count": 2,
        "files": [
            {"name": "a.pdf", "size_bytes": 1000, "extension": ".pdf",
             "supported": True},
            {"name": "b.txt", "size_bytes": 500, "extension": ".txt",
             "supported": True},
        ],
    })
    return agent, inst


def test_call_dispatches_qa_by_default():
    agent, inst = _make_agent_with_mock_instance()
    out = agent.call("What is sugar kelp?")
    assert out == "ANSWER_FROM_QA"
    inst.qa.assert_called_once_with("What is sugar kelp?", history=None)
    inst.sleep_pass.assert_not_called()


def test_call_dispatches_sleep():
    agent, inst = _make_agent_with_mock_instance()
    out = agent.call("/sleep")
    inst.sleep_pass.assert_called_once()
    inst.qa.assert_not_called()
    assert "Sleep pass complete" in out
    assert "merge" in out.lower()


def test_call_dispatches_sleep_pass_variant():
    agent, inst = _make_agent_with_mock_instance()
    agent.call("/sleep pass")
    inst.sleep_pass.assert_called_once()


def test_call_dispatches_ingest_with_filename():
    agent, inst = _make_agent_with_mock_instance()
    out = agent.call("/ingest foo.pdf")
    agent._ingest_file.assert_called_once_with("foo.pdf")
    inst.qa.assert_not_called()
    assert "Ingested" in out


def test_call_ingest_no_arg_lists_files():
    agent, inst = _make_agent_with_mock_instance()
    out = agent.call("/ingest")
    agent._list_raw_files.assert_called_once()
    agent._ingest_file.assert_not_called()
    inst.qa.assert_not_called()
    assert "a.pdf" in out and "b.txt" in out


def test_call_blocked_during_sleep_pass():
    agent, inst = _make_agent_with_mock_instance()
    inst.sleep_pass_running = True
    out = agent.call("/sleep")
    inst.sleep_pass.assert_not_called()
    inst.qa.assert_not_called()
    assert "paused" in out.lower()


def test_call_passes_history_to_qa():
    agent, inst = _make_agent_with_mock_instance()
    history = [{"role": "user", "content": "earlier turn"}]
    agent.call("follow-up?", history=history)
    inst.qa.assert_called_once_with("follow-up?", history=history)
