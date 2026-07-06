import os
import csv
import uuid
import shutil
import hashlib
import threading
import difflib
from datetime import datetime, date
import pandas as pd
import yaml

EXAMPLES_DIR = "examples"
PRIVATE_HASH_FILE = ".daysync_private"  # gitignored; stores the sha256 of the private passcode


def has_private_passcode() -> bool:
    return os.path.exists(PRIVATE_HASH_FILE)


def set_private_passcode(pw: str):
    with open(PRIVATE_HASH_FILE, "w", encoding="utf-8") as f:
        f.write(hashlib.sha256((pw or "").encode()).hexdigest())


def check_private_passcode(pw: str) -> bool:
    """True if pw matches the stored passcode. If none is set yet, the first pw becomes it."""
    if not has_private_passcode():
        set_private_passcode(pw)
        return True
    with open(PRIVATE_HASH_FILE, "r", encoding="utf-8") as f:
        return f.read().strip() == hashlib.sha256((pw or "").encode()).hexdigest()

# OKF bundle version this implementation targets (see okf/SPEC.md v0.1).
OKF_VERSION = "0.1"

# Reserved OKF filenames that are NOT concept documents.
RESERVED_OKF_FILES = {"index.md", "log.md"}

# Define paths relative to workspace
CSV_FILE = "agent_vault.csv"
MEDIA_DIR = "media"
KNOWLEDGE_DIR = "knowledge"

# Thread lock to prevent concurrent write issues in Streamlit
db_lock = threading.Lock()

# CSV Columns
#   due_date/due_time  → when a Todo/Reminder is due (time defaults to 08:00)
#   amount             → parsed value for Expenses
#   needs_review       → True ONLY when a critical detail is missing/ambiguous
#   done               → task completed
#   related            → comma-separated ids of related concepts (OKF cross-links)
HEADERS = [
    "id",
    "timestamp",
    "text_source",
    "transcript",
    "category",
    "summary",
    "due_date",
    "due_time",
    "amount",
    "needs_review",
    "review_reason",
    "review_status",
    "done",
    "related",
    "confidential",
    "audio_path",
]

DEFAULT_DUE_TIME = "08:00"
TIMED_CATEGORIES = ("Todo", "Reminder")


def _normalize_time(t: str) -> str:
    """Return a zero-padded 'HH:MM' string, or '' if unparseable/empty."""
    t = (t or "").strip()
    if not t:
        return ""
    u = t.upper().replace(".", "")
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%I %p", "%I%p", "%H"):
        try:
            return datetime.strptime(u, fmt).strftime("%H:%M")
        except ValueError:
            continue
    return t


def _as_bool(v) -> bool:
    """Coerce a CSV/LLM value (str or bool) into a real bool."""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "1.0", "yes")


def initialize_db():
    """Ensure target directories exist and the CSV database has headers."""
    with db_lock:
        if not os.path.exists(MEDIA_DIR):
            os.makedirs(MEDIA_DIR)
        if not os.path.exists(KNOWLEDGE_DIR):
            os.makedirs(KNOWLEDGE_DIR)
        if not os.path.exists(CSV_FILE):
            with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(HEADERS)
        else:
            _migrate_headers_unsafe()


def _migrate_headers_unsafe():
    """Rewrite the CSV with the current HEADERS if a column was added/removed.
    Keeps existing values, fills new columns with ''. Call within db_lock."""
    with open(CSV_FILE, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    if header == HEADERS:
        return
    with open(CSV_FILE, mode="r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in HEADERS})


def save_task(task_data: dict) -> dict:
    """Appends a new task record to the CSV vault and syncs it to the OKF catalog.
    Returns the complete saved task dict (with id, timestamp, review_status, done)."""
    with db_lock:
        task_id = str(uuid.uuid4())
        # OKF requires ISO 8601 timestamps (local time, no forced 'Z').
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        needs_review = _as_bool(task_data.get("needs_review", False))
        status = "Pending" if needs_review else "Resolved"

        complete_task = {
            "id": task_id,
            "timestamp": timestamp,
            "text_source": task_data.get("text_source", "text"),
            "transcript": (task_data.get("transcript") or "").strip(),
            "category": (task_data.get("category") or "General Note").strip(),
            "summary": (task_data.get("summary") or "").strip(),
            "due_date": (task_data.get("due_date") or "").strip(),
            "due_time": (task_data.get("due_time") or "").strip(),
            "amount": (task_data.get("amount") or "").strip(),
            "needs_review": needs_review,
            "review_reason": (task_data.get("review_reason") or "").strip(),
            "review_status": status,
            "done": False,
            "related": (task_data.get("related") or "").strip(),
            "confidential": _as_bool(task_data.get("confidential", False)),
            "audio_path": (task_data.get("audio_path") or "").strip(),
        }

        with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=HEADERS).writerow(complete_task)

        # Confidential notes are walled off — never written to the OKF bundle / log / index,
        # so they can't leak to the Vault, the Ask agent, or any OKF-aware consumer.
        if not complete_task["confidential"]:
            _sync_to_okf_unsafe(complete_task)
            _append_log_unsafe(complete_task, "Captured")
            _rebuild_index_unsafe()
        return complete_task


def save_and_categorize_task(
    transcript: str,
    category: str,
    summary: str,
    due_date: str = "",
    due_time: str = "",
    amount: str = "",
    needs_review: bool = False,
    review_reason: str = "",
) -> str:
    """Save and categorize a parsed note into the vault.

    Args:
        transcript: Verbatim transcription of the note or the input text.
        category: One of 'Todo', 'Reminder', 'Expense', or 'General Note'.
        summary: A concise, actionable one-sentence summary.
        due_date: For Todo/Reminder — the absolute due date as 'YYYY-MM-DD' (resolve
                  relative dates like 'tomorrow' against today's date). Empty if none.
        due_time: For Todo/Reminder — the time as 'HH:MM' (24h). Empty if unspecified;
                  the app defaults an unspecified time to 08:00.
        amount: For Expense — the numeric amount as a string (e.g. '1500'). Empty otherwise.
        needs_review: True ONLY when a CRITICAL detail is missing or genuinely ambiguous
                      (e.g. an unspecified recipient, item, or amount). Do NOT set this
                      merely because the note has a deadline or involves money.
        review_reason: If needs_review, the specific missing detail to ask the user for.
    """
    import streamlit as st
    text_source = st.session_state.get("current_text_source", "text")
    audio_path = st.session_state.get("current_audio_path", "")

    # Smart default: a dated Todo/Reminder with no explicit time → 08:00.
    if category in TIMED_CATEGORIES and due_date and not due_time:
        due_time = DEFAULT_DUE_TIME
    # Normalize to zero-padded HH:MM (so '8:00' can't be misread as sexagesimal).
    due_time = _normalize_time(due_time)

    saved = save_task({
        "text_source": text_source,
        "transcript": transcript,
        "category": category,
        "summary": summary,
        "due_date": due_date,
        "due_time": due_time,
        "amount": amount,
        "needs_review": needs_review,
        "review_reason": review_reason,
        "audio_path": audio_path,
    })
    st.session_state.last_captured_task = saved
    return f"Task successfully saved with ID: {saved['id']}"


def _read_df() -> pd.DataFrame:
    """Read the CSV into a DataFrame with normalized bool columns."""
    df = pd.read_csv(CSV_FILE, dtype=str).fillna("")
    for col in HEADERS:
        if col not in df.columns:
            df[col] = ""
    df["needs_review_b"] = df["needs_review"].map(_as_bool)
    df["done_b"] = df["done"].map(_as_bool)
    df["confidential_b"] = df["confidential"].map(_as_bool)
    df["review_status"] = df["review_status"].astype(str).str.strip()
    return df


def get_needs_detail() -> pd.DataFrame:
    """Records that need a human to supply a missing detail (needs_review + Pending, not done)."""
    initialize_db()
    try:
        df = _read_df()
        return df[(df["needs_review_b"]) & (df["review_status"] == "Pending") & (~df["done_b"])]
    except Exception as e:
        print(f"Error reading needs-detail queue: {e}")
        return pd.DataFrame(columns=HEADERS)


def get_agenda_items() -> list:
    """Active (not done) Todos/Reminders (plus dated confidential notes) for the agenda."""
    initialize_db()
    try:
        df = _read_df()
        timed = df["category"].isin(TIMED_CATEGORIES)
        dated_private = df["confidential_b"] & (df["due_date"].str.len() > 0)
        active = df[(timed | dated_private) & (~df["done_b"])]
        return active.to_dict("records")
    except Exception as e:
        print(f"Error reading agenda: {e}")
        return []


def get_confidential_items() -> list:
    """All confidential notes as dicts (for the locked Private page)."""
    initialize_db()
    try:
        df = _read_df()
        return df[df["confidential_b"]].to_dict("records")
    except Exception as e:
        print(f"Error reading confidential items: {e}")
        return []


def find_related(task: dict) -> dict:
    """Agentic check: find likely duplicates and time-conflicts among active tasks.

    Compares the given task against other active (not-done) Todos/Reminders:
      - duplicate: same category and a fuzzy summary similarity >= 0.6
      - conflict:  same due_date AND due_time (both set) as a different task
    Returns {'duplicates': [...], 'conflicts': [...]}.
    """
    tid = task.get("id")
    cat = task.get("category")
    summ = (task.get("summary") or "").strip().lower()
    dd = (task.get("due_date") or "").strip()
    dt = (task.get("due_time") or "").strip()
    dups, conf = [], []
    for it in get_agenda_items():
        if it.get("id") == tid:
            continue
        other = {"summary": it.get("summary", ""), "due_date": it.get("due_date", ""),
                 "due_time": it.get("due_time", ""), "category": it.get("category", "")}
        ratio = difflib.SequenceMatcher(None, summ, other["summary"].strip().lower()).ratio()
        if it.get("category") == cat and ratio >= 0.6:
            dups.append({**other, "ratio": round(ratio, 2)})
        elif dd and dt and it.get("due_date") == dd and it.get("due_time") == dt:
            conf.append(other)
    return {"duplicates": dups, "conflicts": conf}


def get_completed_items() -> list:
    """Completed Todos/Reminders as dicts (for the Agenda's done section)."""
    initialize_db()
    try:
        df = _read_df()
        done = df[(df["category"].isin(TIMED_CATEGORIES)) & (df["done_b"])]
        return done.to_dict("records")
    except Exception as e:
        print(f"Error reading completed items: {e}")
        return []


def mark_done(task_id: str, done: bool = True) -> bool:
    """Flip a task's completion state and re-sync its OKF file."""
    with db_lock:
        if not os.path.exists(CSV_FILE):
            return False
        rows, found, updated_row = [], False, None
        with open(CSV_FILE, mode="r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["id"] == task_id:
                    row["done"] = str(bool(done))
                    found = True
                    updated_row = row
                rows.append(row)
        if not found:
            return False
        with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HEADERS)
            w.writeheader()
            w.writerows(rows)
        _normalize_row_bools(updated_row)
        if not _as_bool(updated_row.get("confidential")):
            _sync_to_okf_unsafe(updated_row)
            _append_log_unsafe(updated_row, "Completed" if done else "Reopened")
            _rebuild_index_unsafe()
        return True


def resolve_task(task_id: str, updated_data: dict) -> bool:
    """Apply user corrections, mark the item Resolved, and update its OKF file."""
    editable = ("transcript", "summary", "category", "due_date", "due_time", "amount")
    with db_lock:
        if not os.path.exists(CSV_FILE):
            return False
        rows, found, updated_row = [], False, None
        with open(CSV_FILE, mode="r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["id"] == task_id:
                    for key in editable:
                        if key in updated_data:
                            row[key] = str(updated_data[key]).strip()
                    row["review_status"] = "Resolved"
                    found = True
                    updated_row = row
                rows.append(row)
        if not found:
            return False
        with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HEADERS)
            w.writeheader()
            w.writerows(rows)
        _normalize_row_bools(updated_row)
        _sync_to_okf_unsafe(updated_row)
        _append_log_unsafe(updated_row, "Resolved")
        _rebuild_index_unsafe()
        return True


def resolve_id(ref: str) -> str:
    """Resolve a task reference (full id, id prefix, or summary substring) to a full id.
    Returns '' if nothing matches. Used by the assistant's action tools."""
    ref = (ref or "").strip()
    if not ref:
        return ""
    try:
        df = _read_df()
    except Exception:
        return ""
    ids = list(df["id"])
    if ref in ids:
        return ref
    for i in ids:
        if i.startswith(ref):
            return i
    low = ref.lower()
    for _, r in df.iterrows():
        if low in str(r.get("summary", "")).lower():
            return r["id"]
    return ""


def delete_task(task_id: str) -> bool:
    """Permanently remove a note: its CSV row, OKF concept file, and audio file.
    Rebuilds the index and appends a 'Deleted' entry to the change log."""
    with db_lock:
        if not os.path.exists(CSV_FILE):
            return False
        rows, deleted = [], None
        with open(CSV_FILE, mode="r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["id"] == task_id:
                    deleted = row
                    continue  # drop this row
                rows.append(row)
        if deleted is None:
            return False
        with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HEADERS)
            w.writeheader()
            w.writerows(rows)
        # Remove the OKF concept file
        md_path = os.path.join(KNOWLEDGE_DIR, f"{task_id}.md")
        if os.path.exists(md_path):
            try:
                os.remove(md_path)
            except OSError:
                pass
        # Remove the audio file, if any
        audio = (deleted.get("audio_path") or "").strip()
        if audio and os.path.exists(audio):
            try:
                os.remove(audio)
            except OSError:
                pass
        _normalize_row_bools(deleted)
        if not _as_bool(deleted.get("confidential")):
            _append_log_unsafe(deleted, "Deleted")
            _rebuild_index_unsafe()
        return True


def _normalize_row_bools(row: dict):
    """In-place: convert a CSV row's stringy bool fields into real bools for OKF sync."""
    row["needs_review"] = _as_bool(row.get("needs_review"))
    row["done"] = _as_bool(row.get("done"))


# ── OKF sync helpers ─────────────────────────────────────────────────────────
def _to_iso8601(ts: str) -> str:
    """Normalize a stored timestamp to ISO 8601 (accepts legacy space-separated too)."""
    if not ts:
        return ts
    ts = ts.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(ts, fmt).strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
    return ts


def _build_tags(task: dict) -> list:
    tags = [task["category"].lower().replace(" ", "-"), task["text_source"]]
    if _as_bool(task.get("needs_review")):
        tags.append("needs-detail")
    if _as_bool(task.get("done")):
        tags.append("done")
    return tags


def _sync_to_okf_unsafe(task: dict):
    """Generate/update the OKF v0.1 concept file for a task. Call within db_lock."""
    filepath = os.path.join(KNOWLEDGE_DIR, f"{task['id']}.md")
    iso_ts = _to_iso8601(task["timestamp"])
    needs_review = _as_bool(task.get("needs_review"))
    done = _as_bool(task.get("done"))
    due_date = (task.get("due_date") or "").strip()
    due_time = (task.get("due_time") or "").strip()
    amount = (task.get("amount") or "").strip()
    related_ids = [x for x in (task.get("related") or "").split(",") if x.strip()]

    frontmatter = {
        "type": task["category"],
        "title": task["summary"],
        "description": task["summary"],
    }
    if task["text_source"] == "voice" and task.get("audio_path"):
        frontmatter["resource"] = task["audio_path"]
    frontmatter["tags"] = _build_tags(task)
    frontmatter["timestamp"] = iso_ts
    # Extended (DaySync-specific) fields:
    frontmatter["id"] = task["id"]
    frontmatter["category"] = task["category"]
    if due_date:
        frontmatter["due_date"] = due_date
    if due_time:
        frontmatter["due_time"] = due_time
    if amount:
        frontmatter["amount"] = amount
    frontmatter["needs_review"] = needs_review
    frontmatter["review_reason"] = task.get("review_reason", "")
    frontmatter["review_status"] = task.get("review_status", "Resolved")
    frontmatter["done"] = done
    if related_ids:
        frontmatter["related"] = related_ids
    frontmatter["text_source"] = task["text_source"]
    frontmatter["audio_path"] = task.get("audio_path", "")

    yaml_frontmatter = (
        "---\n"
        + yaml.dump(frontmatter, sort_keys=False, default_flow_style=False, allow_unicode=True)
        + "---\n"
    )

    due_line = ""
    if due_date:
        due_line = f"**Due:** {due_date}" + (f" at {due_time}" if due_time else "") + "\n\n"
    amount_line = f"**Amount:** {amount}\n\n" if amount else ""

    markdown_body = (
        f"# {task['category']}: {task['summary']}\n\n"
        f"**Logged on:** {iso_ts} via *{task['text_source']}*\n\n"
        f"{due_line}{amount_line}"
        "## Verbatim Transcript\n"
        f"{task['transcript']}\n\n"
        "## Structured Summary\n"
        f"{task['summary']}\n"
    )
    if related_ids:
        titles = _title_map_unsafe()
        links = "\n".join(f"- [{titles.get(rid, rid)}](/{rid}.md)" for rid in related_ids)
        markdown_body += f"\n## Related\n{links}\n"
    if needs_review:
        markdown_body += f"\n> [!WARNING]\n> **Needs a detail:** {task.get('review_reason','')}\n"

    with open(filepath, mode="w", encoding="utf-8") as f:
        f.write(yaml_frontmatter + markdown_body)


def load_demo_data() -> bool:
    """Replace the live vault with the curated sample bundle from examples/.
    Copies examples/agent_vault.csv → the CSV and examples/okf-bundle/*.md → knowledge/."""
    src_csv = os.path.join(EXAMPLES_DIR, "agent_vault.csv")
    src_bundle = os.path.join(EXAMPLES_DIR, "okf-bundle")
    if not os.path.exists(src_csv) or not os.path.isdir(src_bundle):
        return False
    initialize_db()
    with db_lock:
        shutil.copyfile(src_csv, CSV_FILE)
        for f in os.listdir(KNOWLEDGE_DIR):
            if f.endswith(".md"):
                os.remove(os.path.join(KNOWLEDGE_DIR, f))
        for f in os.listdir(src_bundle):
            if f.endswith(".md"):
                shutil.copyfile(os.path.join(src_bundle, f), os.path.join(KNOWLEDGE_DIR, f))
        _migrate_headers_unsafe()
    return True


def _title_map_unsafe() -> dict:
    """Return {id: summary} read directly from the CSV (no lock). Call within db_lock."""
    m = {}
    if not os.path.exists(CSV_FILE):
        return m
    with open(CSV_FILE, mode="r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m[r["id"]] = r.get("summary", "")
    return m


def set_related_links(mapping: dict) -> int:
    """Apply concept cross-links. mapping = {id: [related_ids, ...]}.
    Updates the CSV `related` column and re-syncs every concept's OKF file so both
    sides of each link render. Returns the number of concepts given links."""
    with db_lock:
        if not os.path.exists(CSV_FILE):
            return 0
        rows = []
        with open(CSV_FILE, mode="r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        valid_ids = {r["id"] for r in rows}
        linked = 0
        for r in rows:
            rels = [x for x in mapping.get(r["id"], []) if x in valid_ids and x != r["id"]]
            r["related"] = ",".join(rels)
            if rels:
                linked += 1
        with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HEADERS)
            w.writeheader()
            w.writerows(rows)
        for r in rows:
            _normalize_row_bools(r)
            _sync_to_okf_unsafe(r)
        _rebuild_index_unsafe()
        return linked


def _append_log_unsafe(task: dict, action: str):
    """Append an entry to the OKF bundle's reserved log.md. Call within db_lock."""
    log_path = os.path.join(KNOWLEDGE_DIR, "log.md")
    date_heading = f"## {_to_iso8601(task['timestamp'])[:10]}"
    entry = f"- **{action}** — [{task['category']}] {task['summary']} (`{task['id'][:8]}`)\n"
    if os.path.exists(log_path):
        with open(log_path, mode="r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = "# Change Log\n\nChronological history of concepts captured and resolved in this OKF bundle.\n"
    if date_heading not in content:
        content = content.rstrip() + f"\n\n{date_heading}\n\n"
    content = content.rstrip("\n") + "\n" + entry
    with open(log_path, mode="w", encoding="utf-8") as f:
        f.write(content)


def _rebuild_index_unsafe():
    """Rebuild the OKF bundle-root index.md. Call within db_lock."""
    index_path = os.path.join(KNOWLEDGE_DIR, "index.md")
    rows = []
    for file in sorted(os.listdir(KNOWLEDGE_DIR)):
        if not file.endswith(".md") or file in RESERVED_OKF_FILES:
            continue
        try:
            with open(os.path.join(KNOWLEDGE_DIR, file), mode="r", encoding="utf-8") as f:
                content = f.read()
            meta = {}
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    meta = yaml.safe_load(parts[1]) or {}
            rows.append((meta, file))
        except Exception as e:
            print(f"Error indexing OKF file {file}: {e}")

    rows.sort(key=lambda r: str(r[0].get("timestamp", "")), reverse=True)
    lines = [
        "---",
        f'okf_version: "{OKF_VERSION}"',
        "---",
        "# DaySync AI Knowledge Bundle",
        "",
        "Personal concepts captured by DaySync AI, conforming to the "
        "[Open Knowledge Format (OKF)](https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing) v0.1.",
        "",
        f"**Concepts:** {len(rows)}",
        "",
        "## Concepts",
        "",
    ]
    if not rows:
        lines.append("_No concepts captured yet._")
    else:
        for meta, file in rows:
            title = meta.get("title", file)
            ctype = meta.get("type", "Concept")
            ts = meta.get("timestamp", "")
            lines.append(f"- **{ctype}** — [{title}](/{file}) · `{ts}`")
    lines.append("")
    with open(index_path, mode="w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def get_all_okf_concepts() -> list:
    """Scan knowledge/ and parse all OKF concept files (skips reserved files)."""
    initialize_db()
    concepts = []
    if not os.path.exists(KNOWLEDGE_DIR):
        return concepts

    for file in os.listdir(KNOWLEDGE_DIR):
        if not file.endswith(".md") or file in RESERVED_OKF_FILES:
            continue
        filepath = os.path.join(KNOWLEDGE_DIR, file)
        try:
            with open(filepath, mode="r", encoding="utf-8") as f:
                content = f.read()
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter_str, body = parts[1], parts[2].strip()
                    try:
                        frontmatter = yaml.safe_load(frontmatter_str) or {}
                        if not isinstance(frontmatter, dict):
                            raise ValueError("frontmatter is not a mapping")
                    except Exception:
                        frontmatter = {}
                        for line in frontmatter_str.strip().split("\n"):
                            if ":" in line:
                                key, val = line.split(":", 1)
                                val = val.strip().strip('"').strip("'")
                                if val.lower() == "true":
                                    val = True
                                elif val.lower() == "false":
                                    val = False
                                frontmatter[key.strip()] = val
                    if frontmatter.get("timestamp") is not None:
                        frontmatter["timestamp"] = str(frontmatter["timestamp"])
                    concepts.append({"metadata": frontmatter, "body": body,
                                     "filepath": filepath, "filename": file})
        except Exception as e:
            print(f"Error parsing OKF file {file}: {e}")

    concepts.sort(key=lambda x: x["metadata"].get("timestamp", ""), reverse=True)
    return concepts


def get_stats() -> dict:
    """Quick stats for the dashboard."""
    initialize_db()
    try:
        df = _read_df()
        today = date.today().isoformat()
        active = df[(df["category"].isin(TIMED_CATEGORIES)) & (~df["done_b"])]
        dated = active[active["due_date"].str.len() > 0]
        return {
            "total": len(df),
            "needs_detail": int(((df["needs_review_b"]) & (df["review_status"] == "Pending") & (~df["done_b"])).sum()),
            "overdue": int((dated["due_date"] < today).sum()),
            "today": int((dated["due_date"] == today).sum()),
            "active": len(active),
            "done": int(df["done_b"].sum()),
            # kept for backward-compat with any older callers
            "pending": int(((df["needs_review_b"]) & (df["review_status"] == "Pending") & (~df["done_b"])).sum()),
            "resolved": int(len(df) - df["needs_review_b"].sum()),
        }
    except Exception as e:
        print(f"Error computing stats: {e}")
        return {"total": 0, "needs_detail": 0, "overdue": 0, "today": 0,
                "active": 0, "done": 0, "pending": 0, "resolved": 0}
