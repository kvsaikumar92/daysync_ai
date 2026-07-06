"""Regenerate the curated sample OKF bundle in examples/ (a committed showcase).

Run from anywhere:  python scripts/build_examples.py
The live app's own vault/knowledge are untouched — this only writes under examples/.
"""
import os
import sys
import shutil
import tempfile

# Make the project root importable regardless of where this is run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db_helper as db  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "examples")
os.makedirs(BASE, exist_ok=True)
db.CSV_FILE = os.path.join(BASE, "agent_vault.csv")
db.KNOWLEDGE_DIR = os.path.join(BASE, "okf-bundle")
db.MEDIA_DIR = tempfile.mkdtemp()  # samples have no audio; keep examples/ clean

if os.path.exists(db.KNOWLEDGE_DIR):
    shutil.rmtree(db.KNOWLEDGE_DIR)
if os.path.exists(db.CSV_FILE):
    os.remove(db.CSV_FILE)
db.initialize_db()


def mk(**kw):
    d = {"text_source": "text", "needs_review": False, "review_reason": "",
         "due_date": "", "due_time": "", "amount": "", "related": ""}
    d["transcript"] = kw.pop("transcript", kw["summary"])
    d.update(kw)
    return db.save_task(d)["id"]


# Health thread
meds = mk(category="Todo", summary="Buy medicines",
          transcript="Buy medicines tomorrow morning", due_date="2026-07-06", due_time="08:00")
walk = mk(category="General Note", summary="Doctor advised a 30-minute walk daily",
          transcript="Doctor said I should walk for 30 minutes every day")
# Travel thread
tatkal = mk(category="Reminder", summary="Book tatkal ticket for the Chennai trip",
            transcript="Reminder to book the tatkal ticket for Chennai at 9:50 AM",
            due_date="2026-07-06", due_time="09:50")
pack = mk(category="Todo", summary="Pack bags for the Chennai trip",
          transcript="Pack bags for the Chennai trip", due_date="2026-07-06", due_time="18:00")
# Groceries thread (task + its expense)
groc = mk(category="Todo", summary="Buy groceries",
          transcript="Buy groceries today", due_date="2026-07-05", due_time="17:00")
grocexp = mk(category="Expense", summary="Bought groceries",
             transcript="Spent 2400 rupees on groceries", amount="2400")
# Standalone expense + overdue todo + inbox item
mk(category="Expense", summary="Pay the electricity bill",
   transcript="Pay the electricity bill of 1500 rupees before Friday", amount="1500")
mk(category="Todo", summary="Submit the expense report",
   transcript="Submit the expense report", due_date="2026-07-04", due_time="10:00")  # overdue
mk(category="Todo", summary="Pay the bill", transcript="remember to pay the bill",
   needs_review=True, review_reason="Which bill? No amount or due date was specified.")

db.mark_done(meds, True)  # a completed task

db.set_related_links({           # concept cross-links (knowledge graph)
    meds: [walk], walk: [meds],
    tatkal: [pack], pack: [tatkal],
    groc: [grocexp], grocexp: [groc],
})

print("Built examples/ bundle:")
for f in sorted(os.listdir(db.KNOWLEDGE_DIR)):
    print("  ", f)
