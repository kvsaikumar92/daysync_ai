"""Integrity tests for DaySync's CSV vault + OKF bundle.

Runs against a throwaway temp directory, so it never touches your real data.
Run standalone:   python tests/test_db.py
Or with pytest:   pytest tests/
"""
import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db_helper as db  # noqa: E402


def _redirect_to_tmp():
    tmp = tempfile.mkdtemp(prefix="daysync_test_")
    db.CSV_FILE = os.path.join(tmp, "agent_vault.csv")
    db.KNOWLEDGE_DIR = os.path.join(tmp, "knowledge")
    db.MEDIA_DIR = os.path.join(tmp, "media")
    return tmp


def run():
    tmp = _redirect_to_tmp()
    try:
        db.initialize_db()

        # 1. Capture writes CSV + a conformant OKF concept file
        todo = db.save_task({"text_source": "text", "transcript": "Buy milk tomorrow",
                             "category": "Todo", "summary": "Buy milk",
                             "due_date": "2026-07-06", "due_time": "08:00"})
        md = os.path.join(db.KNOWLEDGE_DIR, todo["id"] + ".md")
        assert os.path.exists(md), "concept .md not written"

        concepts = db.get_all_okf_concepts()
        assert len(concepts) == 1, "expected 1 concept"
        assert concepts[0]["metadata"].get("type") == "Todo", "type missing/wrong"
        assert concepts[0]["metadata"].get("due_date") == "2026-07-06", "due_date not synced"

        # 2. Agenda + completion
        assert len(db.get_agenda_items()) == 1, "todo should be in agenda"
        assert db.mark_done(todo["id"]), "mark_done failed"
        assert len(db.get_agenda_items()) == 0, "done todo should leave agenda"
        assert len(db.get_completed_items()) == 1, "completed list should have 1"

        # 3. Needs-a-detail (inbox) + resolve
        amb = db.save_task({"text_source": "text", "transcript": "pay the bill",
                            "category": "Todo", "summary": "Pay the bill",
                            "needs_review": True, "review_reason": "which bill?"})
        assert not db.get_needs_detail().empty, "inbox should have the ambiguous note"
        assert db.resolve_task(amb["id"], {"summary": "Pay electricity bill",
                                           "due_date": "2026-07-07", "due_time": "09:00"}), "resolve failed"
        assert db.get_needs_detail().empty, "inbox should be clear after resolve"

        # 4. Concept cross-linking renders OKF links both ways
        n = db.set_related_links({todo["id"]: [amb["id"]], amb["id"]: [todo["id"]]})
        assert n == 2, "both concepts should be linked"
        body = open(md, encoding="utf-8").read()
        assert "## Related" in body, "Related section missing"
        assert amb["id"] in body, "cross-link to related concept missing"

        # 5. Bundle conformance: every concept has a non-empty type; index declares okf_version
        for c in db.get_all_okf_concepts():
            assert c["metadata"].get("type"), "a concept is missing its required 'type'"
        index = open(os.path.join(db.KNOWLEDGE_DIR, "index.md"), encoding="utf-8").read()
        assert 'okf_version: "0.1"' in index, "index.md must declare okf_version"

        # 6. Delete removes CSV row + OKF file
        assert db.delete_task(todo["id"]), "delete failed"
        assert not os.path.exists(md), "concept .md should be gone after delete"
        assert db.delete_task("does-not-exist") is False, "deleting a missing id should be False"

        print("ALL TESTS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_db_roundtrip():
    """pytest entry point."""
    run()


if __name__ == "__main__":
    run()
