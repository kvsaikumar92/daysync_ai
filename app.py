import os
import re
import json
import time
import uuid
import base64
from datetime import datetime, date
import yaml
import dotenv
import streamlit as st
import pandas as pd
import altair as alt
from google import genai
from google.genai import types

import db_helper

dotenv.load_dotenv()

# ---------------------------------------------------------------------------
# Model Fallback Chain (updated July 2026) — only currently live GA models.
# ---------------------------------------------------------------------------
MODEL_FALLBACK_CHAIN = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.1-flash-lite",
]
_RETRYABLE_SIGNALS = ("503", "429", "UNAVAILABLE", "ResourceExhausted", "quota", "404", "NOT_FOUND")


def get_system_instruction() -> str:
    """System prompt, stamped with today's date so relative dates resolve correctly."""
    today = date.today()
    return (
        "You are the core intelligence of DaySync AI, a smart daily personal concierge agent.\n"
        f"Today's date is {today.isoformat()} ({today.strftime('%A')}).\n"
        "Analyze the user's audio or text note and structure it by calling save_and_categorize_task.\n"
        "1. Transcribe the note verbatim.\n"
        "2. Classify strictly into one of: 'Todo', 'Reminder', 'Expense', or 'General Note'.\n"
        "3. Write a concise, actionable one-sentence summary. Do NOT bake relative time words "
        "(like 'tomorrow') into the summary when you have captured them as a due date.\n"
        "4. For Todo/Reminder: set due_date to an ABSOLUTE 'YYYY-MM-DD' by resolving relative "
        "expressions ('today', 'tomorrow', 'next Friday') against today's date. Set due_time as "
        "'HH:MM' (24-hour) only if a time is stated; otherwise leave it empty (the app defaults to 08:00).\n"
        "5. For Expense: put the numeric amount (digits only) into amount.\n"
        "6. Set needs_review = True ONLY when a CRITICAL detail is missing or genuinely ambiguous so "
        "you cannot confidently file the note — e.g. an unnamed person ('call him'), an unspecified "
        "item/bill, or a payment with no amount. Do NOT set needs_review just because the note has a "
        "deadline or involves money. When True, state the exact missing detail in review_reason."
    )


def call_gemini_with_fallback(client, contents, config):
    last_exc = None
    for model in MODEL_FALLBACK_CHAIN:
        try:
            return client.models.generate_content(model=model, contents=contents, config=config), model
        except Exception as exc:
            if any(sig in str(exc) for sig in _RETRYABLE_SIGNALS):
                last_exc = exc
                continue
            raise
    raise last_exc


# ── Brand assets ─────────────────────────────────────────────────────────────
# Tight-cropped mark (the source logo.png has ~70% transparent padding, so it
# renders muddy at small sizes); fall back to the original if the mark is missing.
LOGO_PATH = "assets/logo_mark.png" if os.path.exists("assets/logo_mark.png") else "assets/logo.png"


@st.cache_data(show_spinner=False)
def _img_b64(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except Exception:
        return ""


st.set_page_config(
    page_title="DaySync AI — Personal Digital Concierge",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Category styling ─────────────────────────────────────────────────────────
CAT_COLORS = {"Todo": "#2563EB", "Reminder": "#7C3AED", "Expense": "#059669", "General Note": "#64748B"}
CAT_BADGE = {"Todo": "badge-todo", "Reminder": "badge-reminder", "Expense": "badge-expense",
             "General Note": "badge-general", "Confidential": "badge-confidential"}
CAT_ICON = {"Todo": "✅", "Reminder": "⏰", "Expense": "💰", "General Note": "📝", "Confidential": "🔒"}


def _fmt_time(hhmm: str) -> str:
    try:
        return datetime.strptime(hhmm.strip(), "%H:%M").strftime("%I:%M %p").lstrip("0")
    except Exception:
        return hhmm


def _fmt_due(due_date: str, due_time: str = "") -> str:
    """Friendly relative due label, e.g. 'Today · 8:00 AM', 'Tue 08 Jul · 5:00 PM'."""
    due_date = (due_date or "").strip()
    if not due_date:
        return ""
    try:
        d = date.fromisoformat(due_date)
    except Exception:
        return due_date
    delta = (d - date.today()).days
    if delta == 0:
        label = "Today"
    elif delta == 1:
        label = "Tomorrow"
    elif delta == -1:
        label = "Yesterday"
    elif 0 < delta < 7:
        label = d.strftime("%A")
    else:
        label = d.strftime("%a %d %b")
    if due_time:
        label += f" · {_fmt_time(due_time)}"
    return label


# ── Design system ────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root {
    --teal:#14B8A6; --teal-deep:#0D9488; --blue:#0EA5E9; --grad:linear-gradient(135deg,#2DD4BF 0%,#38BDF8 100%);
    --grad-soft:linear-gradient(135deg,#ECFEFF 0%,#EFF6FF 55%,#F5F3FF 100%);
    --ink:#0F172A; --body:#475569; --muted:#94A3B8; --line:#E7EDF3; --card:#FFFFFF;
    --shadow:0 6px 24px rgba(15,23,42,0.06); --shadow-teal:0 8px 22px rgba(14,165,233,0.22);
    --warn:#F59E0B; --ok:#10B981; --red:#EF4444;
}
html, body, [class*="css"], .stApp { font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif; }
.stApp { background:
    radial-gradient(1100px 500px at 15% -8%, #E7FBF6 0%, rgba(231,251,246,0) 55%),
    radial-gradient(1000px 520px at 100% 0%, #E9F4FF 0%, rgba(233,244,255,0) 50%), #F6F9FC; }
header[data-testid="stHeader"] { background:transparent; pointer-events:none; }
header[data-testid="stHeader"] [data-testid="stToolbar"] { pointer-events:auto; }
[data-testid="stAppDeployButton"] { display:none; }
.block-container { padding-top:2rem; padding-bottom:3rem; max-width:860px; margin-left:auto; margin-right:auto; }

/* Top nav */
.ds-nav-brand { display:flex; align-items:center; gap:10px; }
.ds-nav-brand img { width:38px; height:38px; border-radius:11px; filter:drop-shadow(0 4px 10px rgba(45,212,191,0.30)); }
.ds-nav-name { font-weight:800; font-size:1.1rem; letter-spacing:-0.5px; white-space:nowrap;
    background:linear-gradient(120deg,#0D9488 0%,#0EA5E9 60%,#6366F1 100%);
    -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
/* Tab bar: content-width tabs, centered, horizontal scroll when they overflow */
.st-key-ds-topnav [data-testid="stHorizontalBlock"] { flex-wrap:nowrap; overflow-x:auto; overflow-y:hidden;
    align-items:center; justify-content:center; gap:4px; }
.st-key-ds-topnav [data-testid="stColumn"] { flex:0 0 auto !important; width:auto !important; min-width:0 !important; }
.st-key-ds-topnav [data-testid="stHorizontalBlock"]::-webkit-scrollbar { height:6px; }
.st-key-ds-topnav [data-testid="stHorizontalBlock"]::-webkit-scrollbar-thumb { background:#CBD5E1; border-radius:6px; }
[data-testid="stPageLink"] a { border-radius:11px; padding:7px 7px !important; font-weight:600 !important;
    font-size:0.86rem !important; color:var(--body) !important; white-space:nowrap; transition:background .15s,color .15s; }
[data-testid="stPageLink"] a:hover { background:#F1FBF9 !important; color:var(--teal-deep) !important; }
[data-testid="stPageLink"] a[aria-current="page"] { background:var(--grad-soft) !important; color:var(--teal-deep) !important; }
.ds-nav-sep { border:none; border-top:1px solid var(--line); margin:10px 0 22px 0; }

/* Top bar: breadcrumb (left) · brand (center) · menu (right) */
.st-key-ds-topbar [data-testid="stHorizontalBlock"] { flex-wrap:nowrap; align-items:center; gap:6px; }
.ds-brand-link { display:flex; align-items:center; justify-content:center; gap:11px; text-decoration:none !important; margin-bottom:6px; }
.ds-brand-link img { width:42px; height:42px; border-radius:12px; filter:drop-shadow(0 4px 10px rgba(45,212,191,0.30)); }
.ds-brand-link .ds-nav-name { font-size:1.35rem; }
.ds-crumb { font-size:0.76rem; font-weight:700; color:var(--muted); text-transform:uppercase; letter-spacing:0.7px; }
.st-key-ds-menu { align-items:flex-end !important; }
.st-key-ds-menu [data-testid="stElementContainer"] { width:auto !important; }
[data-testid="stPopover"] [data-testid="stPageLink"] a { display:block; margin:1px 0; }

/* Floating quick-action buttons (bottom-right, aligned to the 860px content column) */
.st-key-ds-fab { position:fixed; right:max(20px, calc(50% - 430px)); left:auto; bottom:22px;
    z-index:1000; width:auto !important; display:flex; flex-direction:column; align-items:flex-end; gap:12px; }
.st-key-ds-fab [data-testid="stElementContainer"], .st-key-ds-fab [data-testid="stVerticalBlock"] { width:auto !important; }
.st-key-ds-fab [data-testid="stPageLink"] a,
.st-key-ds-fab [data-testid="stPageLink"] a[aria-current="page"] {
    background:var(--grad) !important; color:#fff !important; border-radius:999px !important;
    padding:13px 20px !important; font-weight:700 !important; font-size:0.9rem !important;
    white-space:nowrap; box-shadow:0 10px 26px rgba(14,165,233,0.38) !important;
    transition:transform .15s, filter .15s; }
.st-key-ds-fab [data-testid="stPageLink"] a:hover { transform:translateY(-2px); filter:brightness(1.06); color:#fff !important; }
.st-key-ds-fab [data-testid="stIconMaterial"] { color:#fff !important; }
@media (max-width: 640px) {
    .st-key-ds-fab { right:14px; left:auto; bottom:16px; gap:10px; }
    .st-key-ds-fab [data-testid="stPageLink"] a { padding:12px 16px !important; }
}
[data-testid="stPopover"] button { border-radius:11px !important; border:1px solid var(--line) !important; background:#fff !important;
    color:var(--body) !important; font-weight:600 !important; box-shadow:var(--shadow) !important; white-space:nowrap; }
[data-testid="stPopover"] button:hover { border-color:var(--teal) !important; color:var(--teal-deep) !important; }
[data-testid="stPopover"] button::before { content:"⚙️"; }
/* Settings gear is icon-only (label kept for a11y but hidden) to save tab space */
[data-testid="stPopover"] button [data-testid="stMarkdownContainer"], [data-testid="stPopover"] button p { display:none; }
[data-testid="stPopover"] button { padding:8px 11px !important; }

/* Headings */
.ds-page-title { font-size:1.55rem; font-weight:800; letter-spacing:-0.6px; color:var(--ink); margin:0 0 2px 0; }
.ds-eyebrow { font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; color:var(--teal-deep); margin:2px 0 2px 0; }
.ds-help { font-size:0.9rem; color:var(--body); margin:0 0 20px 0; line-height:1.5; }
.ds-group { font-size:0.9rem; font-weight:700; color:var(--ink); margin:18px 0 8px 2px; display:flex; align-items:center; gap:8px; }
.ds-group .cnt { font-size:0.72rem; color:var(--muted); font-weight:600; }
.ds-group.overdue { color:#DC2626; }

/* Cards */
.ds-card { border:1px solid var(--line); border-radius:18px; padding:22px 24px; background:var(--card); box-shadow:var(--shadow); }
.ds-card-label { font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; color:var(--muted); margin-bottom:6px; }
.ds-card-value { font-size:0.95rem; color:var(--ink); line-height:1.6; }
.ds-card-meta { font-size:0.78rem; color:var(--muted); margin-top:16px; padding-top:14px; border-top:1px solid var(--line); }
.ds-result { border:1px solid var(--line); border-radius:18px; padding:22px 24px; background:#fff; box-shadow:var(--shadow); margin-top:4px; }
.ds-result.warn { border-left:5px solid var(--warn); }
.ds-result.ok { border-left:5px solid var(--ok); }
.ds-result-flag { display:inline-flex; align-items:center; gap:6px; font-size:0.8rem; font-weight:700; }
.ds-result-flag.warn { color:#B45309; } .ds-result-flag.ok { color:#047857; }

/* Task row (agenda) */
.ds-task { display:flex; justify-content:space-between; align-items:center; gap:10px;
    border:1px solid var(--line); border-radius:13px; padding:12px 15px; background:#fff; box-shadow:var(--shadow); }
.ds-task.overdue { border-left:4px solid var(--red); }
.ds-task-title { color:var(--ink); font-weight:600; margin-left:8px; }
.ds-time-chip { display:inline-flex; align-items:center; gap:5px; font-size:0.76rem; font-weight:600;
    color:var(--teal-deep); background:#F0FDFA; border:1px solid #CCFBF1; border-radius:999px; padding:4px 10px; white-space:nowrap; }
.ds-time-chip.overdue { color:#DC2626; background:#FEF2F2; border-color:#FECACA; }
.ds-done .ds-task-title { text-decoration:line-through; color:var(--muted); }
/* keep checkbox + task card on one line, even on mobile */
[class*="st-key-trow-"] [data-testid="stHorizontalBlock"],
[class*="st-key-crow-"] [data-testid="stHorizontalBlock"],
[class*="st-key-prow-"] [data-testid="stHorizontalBlock"] { flex-wrap:nowrap; align-items:center; gap:4px; }
[class*="st-key-trow-"] [data-testid="stColumn"],
[class*="st-key-crow-"] [data-testid="stColumn"],
[class*="st-key-prow-"] [data-testid="stColumn"] { min-width:0 !important; }

/* Inbox card */
.ds-rev { border:1px solid var(--line); border-left:5px solid var(--warn); border-radius:14px 14px 0 0; padding:16px 18px 12px 18px; background:#fff; box-shadow:var(--shadow); }
.ds-rev-reason { font-size:0.82rem; color:#B45309; background:#FFFBEB; border-radius:8px; padding:7px 10px; margin-top:8px; }

.ds-empty { text-align:center; padding:46px 28px; border:1px dashed var(--line); border-radius:18px; background:rgba(255,255,255,0.6); }
.ds-empty h4 { color:var(--ink); margin:0 0 6px 0; font-weight:700; } .ds-empty p { color:var(--muted); margin:0; font-size:0.9rem; }

/* Badges */
.ds-badge { display:inline-block; padding:4px 11px; border-radius:7px; font-size:0.7rem; font-weight:700; letter-spacing:0.4px; text-transform:uppercase; }
.badge-todo{background:#EFF6FF;color:#2563EB;} .badge-reminder{background:#F5F3FF;color:#7C3AED;}
.badge-expense{background:#ECFDF5;color:#059669;} .badge-general{background:#F1F5F9;color:#64748B;}
.badge-urgent{background:#FFF1F2;color:#E11D48;} .badge-confidential{background:#EEF2FF;color:#4F46E5;}

/* Buttons */
.stButton > button, [data-testid="stFormSubmitButton"] button { background:var(--grad) !important; color:#fff !important;
    border:none !important; border-radius:13px !important; font-weight:600 !important; padding:12px 22px !important;
    box-shadow:var(--shadow-teal); transition:transform .15s,filter .15s,box-shadow .15s; }
.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover { transform:translateY(-1px); filter:brightness(1.05);
    box-shadow:0 12px 26px rgba(14,165,233,0.30); color:#fff !important; }
.st-key-ds-save button { font-size:1.05rem !important; padding:16px 22px !important; border-radius:15px !important; }
.st-key-ds-back button, .st-key-ds-discard button, .stButton > button[kind="secondary"] { background:#fff !important;
    color:var(--body) !important; border:1px solid var(--line) !important; box-shadow:var(--shadow) !important; }
.st-key-ds-back button:hover, .st-key-ds-discard button:hover { color:var(--teal-deep) !important; border-color:var(--teal) !important; }
[class*="st-key-delyes"] button { background:#EF4444 !important; color:#fff !important; border:none !important; box-shadow:0 8px 20px rgba(239,68,68,0.30) !important; }
[class*="st-key-delyes"] button:hover { background:#DC2626 !important; color:#fff !important; }

/* Inputs */
.stTextInput input, .stTextArea textarea, .stNumberInput input, .stDateInput input {
    border-radius:12px !important; border:1px solid var(--line) !important; background:#fff !important;
    padding:11px 14px !important; font-size:0.92rem !important; color:var(--ink) !important; }
.stTextInput input:focus, .stTextArea textarea:focus { border-color:var(--teal) !important; box-shadow:0 0 0 3px rgba(45,212,191,0.16) !important; }
[data-baseweb="select"] > div { border-radius:12px !important; border-color:var(--line) !important; }
.stTextInput label, .stTextArea label, .stSelectbox label, .stNumberInput label, .stDateInput label, .stTimeInput label {
    font-weight:600 !important; font-size:0.82rem !important; color:var(--body) !important; }
[data-testid="stSegmentedControl"] button { border-radius:11px !important; font-weight:600 !important; }
[data-testid="stAudioInput"] { max-width:440px; margin:12px auto 0 auto; border:1px dashed #BAE6E0;
    border-radius:18px; background:linear-gradient(135deg,#F0FDFA,#F0F9FF); padding:22px 20px; }
[data-testid="stAudioInput"] [data-testid="stWidgetLabel"] { justify-content:center; text-align:center; }
[data-testid="stAudioInput"] [data-testid="stWidgetLabel"] p { font-weight:600; color:var(--body); font-size:0.95rem; }
/* Turn the built-in record control into a clear, centered Start/Stop pill */
[data-testid="stAudioInput"] [class*="ywdc2n13"] { justify-content:center !important; flex:none !important; }
[data-testid="stAudioInputActionButton"] { display:inline-flex !important; align-items:center; gap:8px;
    background:var(--grad) !important; color:#fff !important; border-radius:999px !important;
    width:auto !important; height:auto !important; padding:12px 24px !important; box-shadow:var(--shadow-teal);
    transition:filter .15s, transform .15s; }
[data-testid="stAudioInputActionButton"] svg { color:#fff !important; }
[data-testid="stAudioInputActionButton"][aria-label*="Record"]::after { content:"Start recording"; font-weight:700; font-size:0.95rem; }
[data-testid="stAudioInputActionButton"][aria-label*="Stop"]::after { content:"Stop"; font-weight:700; font-size:0.95rem; }
[data-testid="stAudioInputActionButton"]:hover { filter:brightness(1.06); transform:translateY(-1px); color:#fff !important; }
/* Save/Discard row: centered under the recorder, matching its width */
.st-key-ds-recbtns { max-width:440px; margin:8px auto 0 auto; }

/* Metrics */
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:16px; padding:16px 18px; box-shadow:var(--shadow); }
[data-testid="stMetricValue"] { font-size:1.6rem !important; font-weight:800; color:var(--ink); }
[data-testid="stMetricLabel"] p { font-size:0.72rem !important; font-weight:600; color:var(--muted) !important; text-transform:uppercase; letter-spacing:0.5px; }

.ds-pill { display:inline-flex; align-items:center; gap:7px; padding:6px 12px; border-radius:999px; font-size:0.78rem; font-weight:600; }
.ds-pill-ok{background:#ECFDF5;color:#059669;} .ds-pill-miss{background:#FFF1F2;color:#E11D48;}
.ds-dot{width:7px;height:7px;border-radius:50%;display:inline-block;} .ds-dot-ok{background:#10B981;} .ds-dot-miss{background:#F43F5E;}
[data-testid="stAlert"] { border-radius:13px; }
[data-testid="stExpander"] details { border:1px solid var(--line) !important; border-radius:14px !important; background:#fff !important; box-shadow:var(--shadow); margin-bottom:10px; }
[data-testid="stExpander"] summary { font-weight:600; }

@media (max-width: 640px) {
    .block-container { padding-left:1rem; padding-right:1rem; padding-top:1.2rem; }
    .ds-page-title { font-size:1.3rem; }
    .ds-card, .ds-result { padding:18px; }
    .ds-brand-link .ds-nav-name { font-size:1.2rem; }
    .st-key-ds-topnav [data-testid="stHorizontalBlock"] { justify-content:flex-start; }
    [data-testid="stPageLink-NavLink"] [data-testid="stMarkdownContainer"] { display:none; }
    [data-testid="stPageLink"] a { padding:9px !important; justify-content:center; }
    [data-testid="stPopover"] button [data-testid="stMarkdownContainer"], [data-testid="stPopover"] button p { display:none; }
    [data-testid="stPopover"] button { padding:9px 12px !important; } [data-testid="stPopover"] button::before { margin-right:0; }
    .st-key-ds-fab [data-testid="stMarkdownContainer"] { display:none; }
    .st-key-ds-kpi [data-testid="stHorizontalBlock"] { flex-wrap:wrap !important; gap:10px !important; }
    .st-key-ds-kpi [data-testid="stColumn"] { flex:1 1 calc(50% - 10px) !important; min-width:calc(50% - 10px) !important; width:calc(50% - 10px) !important; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

LOGO_URI = _img_b64(LOGO_PATH)

db_helper.initialize_db()
st.session_state.setdefault("last_captured_task", None)
st.session_state.setdefault("current_text_source", "text")
st.session_state.setdefault("current_audio_path", "")
st.session_state.setdefault("inbox_selected", None)
st.session_state.setdefault("api_key", os.environ.get("GEMINI_API_KEY", ""))
st.session_state.setdefault("chat", [])
st.session_state.setdefault("digest", "")
st.session_state.setdefault("rec_nonce", 0)
st.session_state.setdefault("confirm_demo", False)
st.session_state.setdefault("link_result", None)
st.session_state.setdefault("private_unlocked", False)
st.session_state.setdefault("priv_nonce", 0)


# ── Capture handler ──────────────────────────────────────────────────────────
def _handle_capture(api_key, source, audio_bytes=None, text_value=""):
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=get_system_instruction(),
        tools=[db_helper.save_and_categorize_task],
    )
    st.session_state.last_captured_task = None
    if source == "voice":
        st.session_state.current_text_source = "voice"
        audio_path = os.path.join(db_helper.MEDIA_DIR, f"voice_{uuid.uuid4()}.wav")
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        st.session_state.current_audio_path = audio_path
        part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
        contents = ["Please analyze and catalog this voice recording:", part]
    else:
        st.session_state.current_text_source = "text"
        st.session_state.current_audio_path = ""
        contents = [f"Please analyze and catalog this text input: {text_value}"]
    _, model_used = call_gemini_with_fallback(client, contents, config)
    return model_used


def _capture_error(e):
    err = str(e)
    if any(sig in err for sig in _RETRYABLE_SIGNALS):
        st.error("❌ All Gemini models are currently busy or rate-limited. Please wait 30–60 seconds and try again.")
        st.info(f"Last error: `{err[:200]}`")
    else:
        st.error(f"❌ Gemini API error: {err}")
        st.info("💡 Verify your API key is correct and the input contains clear content.")


# ── OKF-grounded agents (Ask + Digest) ───────────────────────────────────────
def _okf_context() -> str:
    """Serialize the knowledge/ OKF bundle into compact context for a consumer agent."""
    concepts = db_helper.get_all_okf_concepts()
    if not concepts:
        return "(the knowledge bundle is empty)"
    lines = []
    for c in concepts:
        m = c["metadata"]
        # Completion status is driven ONLY by `done`. `review_status` is a separate
        # human-in-the-loop concept ("Pending" = needs a detail from the user;
        # "Resolved" = no detail needed) and must NOT be reported as task completion.
        if db_helper._as_bool(m.get("done")):
            status = "completed"
        elif str(m.get("review_status", "")).strip().lower() == "pending":
            status = "not done (awaiting a detail from the user)"
        else:
            status = "not done (active)"
        due = (str(m.get("due_date", "")) + " " + str(m.get("due_time", ""))).strip()
        meta = f"[{m.get('category','')}] {m.get('title','')}"
        if due:
            meta += f" | due {due}"
        if m.get("amount"):
            meta += f" | amount {m.get('amount')}"
        meta += f" | status {status} | id {str(m.get('id',''))[:8]}"
        body = " ".join(c["body"].split())[:280]
        lines.append(f"- {meta}\n  {body}")
    return "\n".join(lines)


# ── Action tools the assistant can call to mutate the vault ───────────────────
def complete_task(task_id: str) -> str:
    """Mark a todo or reminder as done/completed.

    Args:
        task_id: The note's id (or a short id prefix) shown in the knowledge bundle.
    """
    tid = db_helper.resolve_id(task_id)
    if not tid:
        return f"No task found matching '{task_id}'."
    return "Marked done." if db_helper.mark_done(tid, True) else "Could not mark it done."


def reschedule_task(task_id: str, due_date: str = "", due_time: str = "") -> str:
    """Change when a task is due.

    Args:
        task_id: The note's id (or a short id prefix).
        due_date: New due date as an absolute 'YYYY-MM-DD' (resolve relative dates against today).
        due_time: New due time as 'HH:MM' (24-hour). Optional.
    """
    tid = db_helper.resolve_id(task_id)
    if not tid:
        return f"No task found matching '{task_id}'."
    data = {}
    if due_date:
        data["due_date"] = due_date
    if due_time:
        data["due_time"] = db_helper._normalize_time(due_time)
    if not data:
        return "Nothing to change — provide a new date or time."
    return "Rescheduled." if db_helper.resolve_task(tid, data) else "Could not reschedule."


def add_task(category: str, summary: str, due_date: str = "", due_time: str = "", amount: str = "") -> str:
    """Create a new note/task in the vault.

    Args:
        category: One of 'Todo', 'Reminder', 'Expense', or 'General Note'.
        summary: A concise one-sentence summary of the task.
        due_date: For Todo/Reminder — absolute 'YYYY-MM-DD'. Optional.
        due_time: 'HH:MM' 24-hour. Optional (defaults to 08:00 for dated Todos/Reminders).
        amount: For Expense — numeric amount. Optional.
    """
    cat = category if category in ("Todo", "Reminder", "Expense", "General Note") else "Todo"
    if cat in ("Todo", "Reminder") and due_date and not due_time:
        due_time = "08:00"
    saved = db_helper.save_task({
        "text_source": "text", "transcript": summary, "category": cat, "summary": summary,
        "due_date": due_date, "due_time": db_helper._normalize_time(due_time),
        "amount": amount, "needs_review": False, "review_reason": "",
    })
    extra = f" (due {saved['due_date']} {saved['due_time']})".rstrip() if saved.get("due_date") else ""
    return f"Added {cat}: {summary}{extra}."


def remove_task(task_id: str) -> str:
    """Permanently delete a note/task from the vault and OKF bundle.

    Args:
        task_id: The note's id (or a short id prefix).
    """
    tid = db_helper.resolve_id(task_id)
    if not tid:
        return f"No task found matching '{task_id}'."
    return "Deleted." if db_helper.delete_task(tid) else "Could not delete."


ASSISTANT_TOOLS = [complete_task, reschedule_task, add_task, remove_task]


def ask_okf(api_key: str, question: str, history: list) -> str:
    """Consumer agent over the OKF bundle that can both ANSWER and ACT (via tools)."""
    client = genai.Client(api_key=api_key)
    ctx = _okf_context()
    convo = "\n".join(f"{h['role']}: {h['content']}" for h in history[-6:])
    sys = (
        "You are DaySync's assistant — a second agent over the user's personal knowledge, captured "
        "by DaySync in Google's Open Knowledge Format (OKF).\n"
        f"Today is {date.today().isoformat()} ({date.today().strftime('%A')}).\n"
        "You can ANSWER questions about the notes AND take actions when the user asks, using tools: "
        "complete_task, reschedule_task, add_task, remove_task. Target actions using the `id` shown "
        "for each item in the bundle. Resolve relative dates ('tomorrow', 'Friday') against today. "
        "Only act when the user clearly asks to; otherwise just answer. After acting, confirm briefly "
        "what you did. Answer strictly from the bundle; if something isn't noted, say so. Be concise.\n"
        "STATUS RULES: each item's `status` is the source of truth. 'completed' = the task is done; "
        "'not done (active)' and 'not done (awaiting a detail from the user)' both mean it is NOT done "
        "and is still pending. Never describe a 'not done' item as done, resolved, or finished."
    )
    config = types.GenerateContentConfig(system_instruction=sys, tools=ASSISTANT_TOOLS)
    contents = [f"KNOWLEDGE BUNDLE (OKF):\n{ctx}\n\nConversation so far:\n{convo}\n\nUser message: {question}"]
    resp, _ = call_gemini_with_fallback(client, contents, config)
    return (resp.text or "").strip() or "Done."


def generate_digest(api_key: str) -> str:
    """An analyst agent that summarizes the whole OKF bundle into a friendly digest."""
    client = genai.Client(api_key=api_key)
    ctx = _okf_context()
    sys = (
        "You are DaySync's analyst agent. Write a short, friendly digest of the user's captured notes, "
        "reading only the OKF knowledge bundle provided.\n"
        f"Today is {date.today().isoformat()}.\n"
        "Cover: how many todos are done vs still pending, anything overdue, the total of all expense "
        "amounts (sum the numbers), and 1–2 highlights. Use tight markdown with a few bullet points. "
        "Keep it under 120 words. Do not invent anything not in the notes."
    )
    config = types.GenerateContentConfig(system_instruction=sys)
    resp, _ = call_gemini_with_fallback(client, [f"Knowledge bundle:\n{ctx}"], config)
    return (resp.text or "").strip()


def link_related_concepts(api_key: str):
    """An agent that connects related notes into a knowledge graph using OKF cross-links.
    Returns (linked_concept_count, [(titleA, titleB), ...] unique pairs)."""
    concepts = db_helper.get_all_okf_concepts()
    if len(concepts) < 2:
        return 0, []
    id2title = {c["metadata"].get("id"): c["metadata"].get("title", "") for c in concepts}
    lines = [f"{c['metadata'].get('id')}: [{c['metadata'].get('category')}] "
             f"{c['metadata'].get('title')} — {c['metadata'].get('description', '')}" for c in concepts]
    ctx = "\n".join(lines)
    client = genai.Client(api_key=api_key)
    sys = (
        "You connect a user's personal notes into a small knowledge graph.\n"
        "Each line is 'id: [category] title — summary'. Identify which notes are GENUINELY related — "
        "same topic, same trip/event, a task and its matching expense, or a clear follow-up. "
        "Return ONLY a JSON object mapping each note id to an array of related note ids. "
        "Use only the ids given. Omit weak or spurious links; empty arrays are fine."
    )
    config = types.GenerateContentConfig(system_instruction=sys, response_mime_type="application/json")
    resp, _ = call_gemini_with_fallback(client, [ctx], config)
    try:
        mapping = json.loads(resp.text)
        if not isinstance(mapping, dict):
            return 0, []
    except Exception:
        return 0, []
    clean = {k: [str(x) for x in v] for k, v in mapping.items() if isinstance(v, (list, tuple))}
    n = db_helper.set_related_links(clean)
    # unique title pairs for a human-readable summary
    seen, pairs = set(), []
    for a, rels in clean.items():
        for b in rels:
            if a == b or a not in id2title or b not in id2title:
                continue
            key = tuple(sorted([a, b]))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((id2title[a], id2title[b]))
    return n, pairs


# ═══════════════════════ PAGE: OVERVIEW ══════════════════════════════════════
def render_overview():
    st.markdown('<p class="ds-eyebrow">Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="ds-page-title">Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="ds-help">Your day at a glance.</p>', unsafe_allow_html=True)

    stats = db_helper.get_stats()
    concepts = db_helper.get_all_okf_concepts()

    with st.container(key="ds-kpi"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Due today", stats["today"])
        c2.metric("Overdue", stats["overdue"])
        c3.metric("Needs a detail", stats["needs_detail"])
        c4.metric("Total notes", stats["total"])

    if not concepts:
        st.markdown('<div class="ds-empty" style="margin-top:22px;"><h4>Nothing captured yet</h4>'
                    '<p>Head to <strong>Capture</strong> to log your first voice or text note.</p></div>',
                    unsafe_allow_html=True)
        return

    # Up next — overdue + today's agenda
    agenda = db_helper.get_agenda_items()
    groups = _agenda_groups(agenda)
    up_next = groups["Overdue"] + groups["Today"] + groups["Tomorrow"]
    if up_next:
        st.markdown('<p class="ds-eyebrow" style="margin-top:12px;">Up next</p>', unsafe_allow_html=True)
        for it in up_next[:4]:
            _task_line(it, overdue=(it in groups["Overdue"]), interactive=False)
        st.markdown('<div style="height:6px;"></div>', unsafe_allow_html=True)

    # By category chart
    counts = {}
    for c in concepts:
        cat = c["metadata"].get("category", "General Note")
        counts[cat] = counts.get(cat, 0) + 1
    df = pd.DataFrame({"category": list(counts.keys()), "count": list(counts.values())})
    st.markdown('<p class="ds-eyebrow" style="margin-top:14px;">By category</p>', unsafe_allow_html=True)
    bars = alt.Chart(df).mark_bar(cornerRadius=5, size=26).encode(
        x=alt.X("count:Q", title=None, axis=alt.Axis(grid=False, tickMinStep=1, labelColor="#94A3B8")),
        y=alt.Y("category:N", sort="-x", title=None, axis=alt.Axis(labelColor="#475569", labelFontSize=13)),
        color=alt.Color("category:N", scale=alt.Scale(domain=list(CAT_COLORS), range=list(CAT_COLORS.values())), legend=None),
        tooltip=[alt.Tooltip("category:N", title="Category"), alt.Tooltip("count:Q", title="Notes")],
    )
    labels = bars.mark_text(align="left", dx=6, color="#475569", fontWeight=600).encode(text="count:Q")
    st.altair_chart((bars + labels).properties(height=max(120, 46 * len(df))), use_container_width=True)

    # Weekly digest — an analyst agent summarizing the OKF bundle
    st.markdown('<p class="ds-eyebrow" style="margin-top:16px;">Weekly digest</p>', unsafe_allow_html=True)
    api_key = st.session_state.get("api_key", "")
    dcol1, dcol2 = st.columns([1, 3], vertical_alignment="center")
    with dcol1:
        gen = st.button("✨ Generate", key="gen_digest", width="stretch")
    with dcol2:
        st.markdown('<span style="font-size:0.82rem;color:#94A3B8;">An agent reads your OKF bundle and writes a recap.</span>',
                    unsafe_allow_html=True)
    if gen:
        if not api_key:
            st.error("Add your API key in ⚙️ Settings first.")
        else:
            with st.spinner("Reading your knowledge bundle…"):
                try:
                    st.session_state.digest = generate_digest(api_key)
                except Exception as e:
                    st.session_state.digest = f"⚠️ {e}"
    if st.session_state.get("digest"):
        with st.container(border=True):
            st.markdown(st.session_state.digest)


# ═══════════════════════ PAGE: CAPTURE ═══════════════════════════════════════
def render_capture():
    api_key = st.session_state.get("api_key", "")
    st.markdown('<p class="ds-eyebrow">Capture</p>', unsafe_allow_html=True)
    st.markdown('<p class="ds-page-title">Log a note</p>', unsafe_allow_html=True)
    st.markdown('<p class="ds-help">Speak or type — Gemini transcribes, classifies, sets a due date, and files it.</p>',
                unsafe_allow_html=True)

    if not api_key:
        st.warning("Add your Gemini API Key to get started — open **⚙️ Settings** in the top-right.")

    mode = st.segmented_control("Input mode", ["🎙️ Voice", "⌨️ Text", "🔒 Private"], default="🎙️ Voice", key="capture_mode")

    if mode == "🎙️ Voice":
        # The recorder itself handles Tap-to-record → Stop; Save/Discard appear after.
        audio_file = st.audio_input("Tap to record your note", key=f"rec_{st.session_state.rec_nonce}")
        if audio_file is not None:
            with st.container(key="ds-recbtns"):   # centered under the recorder, equal widths
                cs, cd = st.columns(2)
                with cs:
                    save_clicked = st.button("💾 Save", type="primary", width="stretch", key="ds-save")
                with cd:
                    discard_clicked = st.button("Discard", type="secondary", width="stretch", key="ds-discard")
            if discard_clicked:
                st.session_state.rec_nonce += 1        # reset the recorder
                st.session_state.last_captured_task = None
                st.rerun()
            if save_clicked:
                if not api_key:
                    st.error("🔑 API Key is required. Open the ☰ Menu and add it.")
                else:
                    with st.spinner("Processing voice note..."):
                        try:
                            mu = _handle_capture(api_key, "voice", audio_bytes=audio_file.read())
                            if mu != MODEL_FALLBACK_CHAIN[0]:
                                st.toast(f"⚡ Primary model busy — processed with {mu}", icon="⚠️")
                            st.session_state.rec_nonce += 1  # clear recorder for the next note
                            st.rerun()
                        except Exception as e:
                            _capture_error(e)
        else:
            st.caption("Tap the mic to record · stop when done · then Save.")
    elif mode == "⌨️ Text":
        text_input = st.text_input("Type your note", placeholder="e.g. Call the dentist Friday at 3pm")
        with st.container(key="ds-save"):
            save_clicked = st.button("💾 Save to Vault", type="primary", width="stretch")
        if save_clicked:
            if not api_key:
                st.error("🔑 API Key is required. Open the ☰ Menu and add it.")
            elif text_input.strip():
                with st.spinner("Processing text note..."):
                    try:
                        mu = _handle_capture(api_key, "text", text_value=text_input.strip())
                        if mu != MODEL_FALLBACK_CHAIN[0]:
                            st.toast(f"⚡ Primary model busy — processed with {mu}", icon="⚠️")
                        st.rerun()
                    except Exception as e:
                        _capture_error(e)
            else:
                st.warning("Please enter a note first.")
    else:  # 🔒 Private — stored locally, never sent to the AI, never in the OKF bundle
        if st.session_state.pop("private_saved", False):
            st.success("🔒 Saved to your Private notes — open the **☰ Menu → Private** page to view them.")
        st.caption("🔒 Private notes stay on your device, are never sent to the AI, and are hidden behind "
                   "your passcode (unlock on the **Private** page).")
        n = st.session_state.priv_nonce  # bump on save to reset the form fields
        ptext = st.text_input("Private note", placeholder="e.g. Locker code is 4821", key=f"priv_note_{n}")
        pc1, pc2 = st.columns(2)
        pdate = pc1.date_input("Remind me on (optional)", value=None, key=f"priv_date_{n}")
        ptime = pc2.time_input("Time (optional)", value=None, key=f"priv_time_{n}")
        with st.container(key="ds-save-private"):
            save_priv = st.button("🔒 Save privately", type="primary", width="stretch")
        if save_priv:
            if ptext.strip():
                db_helper.save_task({
                    "text_source": "text", "transcript": ptext.strip(), "category": "Confidential",
                    "summary": ptext.strip(),
                    "due_date": pdate.isoformat() if pdate else "",
                    "due_time": ptime.strftime("%H:%M") if ptime else "",
                    "confidential": True, "needs_review": False, "review_reason": "",
                })
                st.session_state.last_captured_task = None  # never surface private content in the result card
                st.session_state.private_saved = True
                st.session_state.priv_nonce += 1           # clear the form for the next note
                st.rerun()
            else:
                st.warning("Type a private note first.")

    task = st.session_state.last_captured_task
    if task:
        needs = bool(task["needs_review"])
        variant = "warn" if needs else "ok"
        badge_cls = CAT_BADGE.get(task["category"], "badge-general")
        flag = ('<span class="ds-result-flag warn">⚠️ Needs a detail</span>' if needs
                else '<span class="ds-result-flag ok">✅ Filed</span>')
        due = _fmt_due(task.get("due_date", ""), task.get("due_time", ""))
        due_html = f'<div class="ds-card-label" style="margin-top:14px;">Due</div><div class="ds-card-value">🗓️ {due}</div>' if due else ""
        amt = task.get("amount", "")
        amt_html = f'<div class="ds-card-label" style="margin-top:14px;">Amount</div><div class="ds-card-value">💰 {amt}</div>' if amt else ""
        st.markdown(
            f"""
            <div class="ds-result {variant}">
              <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:14px;">
                <span class="ds-badge {badge_cls}">{task['category']}</span>{flag}
              </div>
              <div class="ds-card-label">Summary</div>
              <div class="ds-card-value" style="font-weight:600;">{task['summary']}</div>
              {due_html}{amt_html}
              <div class="ds-card-label" style="margin-top:14px;">Transcript</div>
              <div class="ds-card-value" style="color:#475569;border-left:3px solid #2DD4BF;padding-left:12px;">{task['transcript']}</div>
              <div class="ds-card-meta">Source: {task['text_source'].capitalize()} · {task['timestamp']} · ID <code>{task['id'][:8]}</code></div>
            </div>
            """, unsafe_allow_html=True)
        if task["text_source"] == "voice" and task["audio_path"] and os.path.exists(task["audio_path"]):
            st.audio(task["audio_path"])
        if needs:
            st.info(f"**Needs a detail:** {task['review_reason']}")
            if st.button("Fix it in Inbox →", type="primary", key="goto_inbox"):
                st.session_state.inbox_selected = task["id"]
                st.switch_page(inbox_page)

        # Agentic dedupe / conflict check against existing active tasks
        if task["category"] in ("Todo", "Reminder"):
            rel = db_helper.find_related(task)
            notes = []
            for d in rel["duplicates"]:
                due = _fmt_due(d.get("due_date", ""), d.get("due_time", ""))
                notes.append(f"Possible **duplicate** of _{d['summary']}_" + (f" (due {due})" if due else ""))
            for c in rel["conflicts"]:
                notes.append(f"**Time clash** with _{c['summary']}_ at {_fmt_time(c.get('due_time',''))}")
            if notes:
                st.warning("🧭 **The agent noticed:**\n\n" + "\n\n".join(f"- {n}" for n in notes))


# ═══════════════════════ AGENDA HELPERS ══════════════════════════════════════
def _agenda_groups(items):
    today = date.today()
    g = {"Overdue": [], "Today": [], "Tomorrow": [], "Upcoming": [], "Someday": []}
    for it in items:
        dd = (it.get("due_date") or "").strip()
        if not dd:
            g["Someday"].append(it); continue
        try:
            d = date.fromisoformat(dd)
        except Exception:
            g["Someday"].append(it); continue
        delta = (d - today).days
        if delta < 0:
            g["Overdue"].append(it)
        elif delta == 0:
            g["Today"].append(it)
        elif delta == 1:
            g["Tomorrow"].append(it)
        else:
            g["Upcoming"].append(it)
    for k in g:
        g[k].sort(key=lambda x: ((x.get("due_date") or ""), (x.get("due_time") or "")))
    return g


def _is_confidential(it) -> bool:
    return str(it.get("confidential", "")).strip().lower() in ("true", "1", "yes")


def _task_line(it, overdue=False, interactive=True):
    """Render one agenda task row (optionally with a done checkbox)."""
    # Mask confidential notes on the agenda unless the private vault is unlocked
    locked = _is_confidential(it) and not st.session_state.get("private_unlocked")
    cat = "Confidential" if _is_confidential(it) else it.get("category", "Todo")
    title = "🔒 Private note" if locked else it.get("summary", "")
    due = _fmt_due(it.get("due_date", ""), it.get("due_time", ""))
    chip = f'<span class="ds-time-chip {"overdue" if overdue else ""}">🕑 {due}</span>' if due else '<span class="ds-time-chip" style="color:#94A3B8;background:#F8FAFC;border-color:#E7EDF3;">No date</span>'
    card = (f'<div class="ds-task {"overdue" if overdue else ""}">'
            f'<div><span class="ds-badge {CAT_BADGE.get(cat,"badge-general")}">{cat}</span>'
            f'<span class="ds-task-title">{title}</span></div>{chip}</div>')
    if not interactive:
        st.markdown(card, unsafe_allow_html=True)
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
        return
    with st.container(key=f"trow-{it['id']}"):
        c1, c2 = st.columns([0.12, 0.88], vertical_alignment="center")
        with c1:
            if st.checkbox("done", key=f"done_{it['id']}", label_visibility="collapsed"):
                db_helper.mark_done(it["id"], True)
                st.rerun()
        with c2:
            st.markdown(card, unsafe_allow_html=True)
    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)


# ═══════════════════════ PAGE: AGENDA ════════════════════════════════════════
def render_agenda():
    st.markdown('<p class="ds-eyebrow">Schedule</p>', unsafe_allow_html=True)
    st.markdown('<p class="ds-page-title">Agenda</p>', unsafe_allow_html=True)
    st.markdown('<p class="ds-help">Your todos and reminders, organized by when they are due. Check one off when done.</p>',
                unsafe_allow_html=True)

    items = db_helper.get_agenda_items()
    groups = _agenda_groups(items)
    if not items:
        st.markdown('<div class="ds-empty"><h4>No active tasks</h4>'
                    '<p>Capture a todo or reminder and it will show up here on its due date.</p></div>',
                    unsafe_allow_html=True)
    else:
        labels = {"Overdue": "⚠️ Overdue", "Today": "Today", "Tomorrow": "Tomorrow",
                  "Upcoming": "Upcoming", "Someday": "Someday"}
        for key in ["Overdue", "Today", "Tomorrow", "Upcoming", "Someday"]:
            bucket = groups[key]
            if not bucket:
                continue
            cls = "ds-group overdue" if key == "Overdue" else "ds-group"
            st.markdown(f'<div class="{cls}">{labels[key]} <span class="cnt">· {len(bucket)}</span></div>',
                        unsafe_allow_html=True)
            for it in bucket:
                _task_line(it, overdue=(key == "Overdue"))

    done_items = db_helper.get_completed_items()
    if done_items:
        with st.expander(f"✓ Completed ({len(done_items)})"):
            for it in done_items:
                with st.container(key=f"crow-{it['id']}"):
                    cc1, cc2 = st.columns([0.12, 0.88], vertical_alignment="center")
                    with cc1:
                        if not st.checkbox("reopen", value=True, key=f"reopen_{it['id']}", label_visibility="collapsed"):
                            db_helper.mark_done(it["id"], False)
                            st.rerun()
                    with cc2:
                        st.markdown(f'<div class="ds-task ds-done"><div>'
                                f'<span class="ds-badge {CAT_BADGE.get(it.get("category","Todo"),"badge-general")}">{it.get("category","Todo")}</span>'
                                f'<span class="ds-task-title">{it.get("summary","")}</span></div></div>',
                                unsafe_allow_html=True)
                    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)


# ═══════════════════════ PAGE: INBOX (needs a detail) ════════════════════════
def render_inbox():
    st.markdown('<p class="ds-eyebrow">Needs your input</p>', unsafe_allow_html=True)
    st.markdown('<p class="ds-page-title">Inbox</p>', unsafe_allow_html=True)

    pending = db_helper.get_needs_detail()
    if pending.empty:
        st.markdown('<div class="ds-empty"><h4>✅ All clear</h4>'
                    '<p>Nothing needs a detail from you. DaySync files clear notes automatically.</p></div>',
                    unsafe_allow_html=True)
        st.session_state.inbox_selected = None
        return

    rows = {r["id"]: r for _, r in pending.iterrows()}
    selected = st.session_state.inbox_selected
    if selected not in rows:
        selected = None

    if selected:
        row = rows[selected]
        if st.button("← Back to inbox", key="ds-back"):
            st.session_state.inbox_selected = None
            st.rerun()
        st.markdown(f'<p class="ds-eyebrow" style="margin-top:10px;">Add the missing detail · {selected[:8]}</p>',
                    unsafe_allow_html=True)
        st.warning(f"🚨 **What's missing:** {row['review_reason']}")

        if row["text_source"] == "voice" and row["audio_path"] and os.path.exists(row["audio_path"]):
            st.markdown("**Original audio recording:**")
            st.audio(row["audio_path"])

        with st.form("resolve_form", clear_on_submit=False):
            c1, c2 = st.columns([2, 1])
            with c1:
                et = st.text_area("Transcript (correct errors)", value=row["transcript"], height=120)
                es = st.text_area("Summary (single sentence)", value=row["summary"], height=70)
            with c2:
                cats = ["Todo", "Reminder", "Expense", "General Note"]
                idx = cats.index(row["category"]) if row["category"] in cats else 0
                ec = st.selectbox("Category", cats, index=idx)
                edate = st.text_input("Due date (YYYY-MM-DD)", value=row.get("due_date", ""))
                etime = st.text_input("Due time (HH:MM)", value=row.get("due_time", ""))
                eamt = st.text_input("Amount", value=row.get("amount", ""))
            if st.form_submit_button("Confirm & File ✅", type="primary", width="stretch"):
                ok = db_helper.resolve_task(selected, {
                    "transcript": et, "summary": es, "category": ec,
                    "due_date": edate, "due_time": etime, "amount": eamt,
                })
                if ok:
                    if st.session_state.last_captured_task and st.session_state.last_captured_task["id"] == selected:
                        st.session_state.last_captured_task = None
                    st.session_state.inbox_selected = None
                    st.success("🎉 Filed! Updated in the vault and synced to OKF.")
                    time.sleep(1.0)
                    st.rerun()
                else:
                    st.error("Error saving your changes.")
        return

    st.markdown(f'<p class="ds-help">{len(rows)} note(s) where DaySync needs a missing detail to file confidently. '
                f'Add it and confirm.</p>', unsafe_allow_html=True)
    for tid, row in rows.items():
        st.markdown(
            f'<div class="ds-rev"><div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">'
            f'<span class="ds-badge {CAT_BADGE.get(row["category"],"badge-general")}">{row["category"]}</span>'
            f'<span style="color:var(--muted);font-size:0.75rem;">{row["timestamp"]}</span></div>'
            f'<div style="color:var(--ink);font-weight:600;margin-top:8px;">{CAT_ICON.get(row["category"],"📝")} {row["summary"]}</div>'
            f'<div class="ds-rev-reason">⚠️ {row["review_reason"]}</div></div>', unsafe_allow_html=True)
        if st.button("Add detail →", key=f"inbox_{tid}", width="stretch"):
            st.session_state.inbox_selected = tid
            st.rerun()
        st.markdown('<div style="height:12px;"></div>', unsafe_allow_html=True)


# ═══════════════════════ PAGE: ASK (OKF-grounded agent) ══════════════════════
def render_ask():
    api_key = st.session_state.get("api_key", "")
    st.markdown('<p class="ds-eyebrow">Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="ds-page-title">Ask your day</p>', unsafe_allow_html=True)
    st.markdown('<p class="ds-help">A second agent over your OKF bundle — it <strong>answers</strong> questions and can '
                '<strong>act</strong> on your notes: complete, reschedule, add, or delete tasks by just asking.</p>',
                unsafe_allow_html=True)

    if not api_key:
        st.warning("Add your Gemini API Key to chat — open the **☰ Menu** in the top-right.")

    if not st.session_state.chat:
        st.markdown('<p style="font-size:0.82rem;color:#94A3B8;">Try: <em>“What\'s due tomorrow?”</em> · '
                    '<em>“Add a todo to call mom tomorrow 6pm”</em> · <em>“Mark buy groceries done”</em></p>',
                    unsafe_allow_html=True)
        cc = st.columns(3)
        starters = ["What's due tomorrow?", "Add a todo to call mom tomorrow 6pm", "Mark buy groceries done"]
        picked = None
        for i, s in enumerate(starters):
            if cc[i].button(s, key=f"starter_{i}", width="stretch"):
                picked = s
        if picked:
            st.session_state._pending_q = picked
            st.rerun()

    for m in st.session_state.chat:
        st.chat_message(m["role"]).markdown(m["content"])

    q = st.chat_input("Ask about your notes…")
    q = q or st.session_state.pop("_pending_q", None)
    if q:
        if not api_key:
            st.error("Add your API key in ⚙️ Settings first.")
        else:
            st.session_state.chat.append({"role": "user", "content": q})
            st.chat_message("user").markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("Reading your knowledge bundle…"):
                    try:
                        ans = ask_okf(api_key, q, st.session_state.chat)
                    except Exception as e:
                        ans = f"⚠️ {e}"
                    st.markdown(ans)
            st.session_state.chat.append({"role": "assistant", "content": ans})


# ═══════════════════════ PAGE: VAULT (OKF) ═══════════════════════════════════
def render_vault():
    st.markdown('<p class="ds-eyebrow">Open Knowledge Format</p>', unsafe_allow_html=True)
    st.markdown('<p class="ds-page-title">Knowledge vault</p>', unsafe_allow_html=True)
    st.markdown('<p class="ds-help">Every note is mirrored as an OKF v0.1 Markdown concept in the '
                '<code>knowledge/</code> bundle — portable and readable by any OKF-aware agent.</p>', unsafe_allow_html=True)

    concepts = db_helper.get_all_okf_concepts()
    if not concepts:
        st.markdown('<div class="ds-empty"><h4>No knowledge concepts synced yet</h4>'
                    '<p>Items you log convert to OKF Markdown and appear here.</p></div>', unsafe_allow_html=True)
        return

    # Agentic cross-linking — connect related concepts into a knowledge graph
    api_key = st.session_state.get("api_key", "")
    lc1, lc2 = st.columns([1, 2], vertical_alignment="center")
    with lc1:
        link_btn = st.button("🔗 Link related", key="link_concepts", width="stretch")
    with lc2:
        st.markdown('<span style="font-size:0.82rem;color:#94A3B8;">An agent links related notes with OKF cross-links.</span>',
                    unsafe_allow_html=True)
    if link_btn:
        if not api_key:
            st.error("Add your API key in the ☰ Menu first.")
        else:
            with st.spinner("Finding connections across your notes…"):
                try:
                    n, pairs = link_related_concepts(api_key)
                    st.session_state.link_result = {"n": n, "pairs": pairs}
                    st.rerun()
                except Exception as e:
                    st.error(f"Linking failed: {e}")

    # Result summary from the last linking pass
    res = st.session_state.get("link_result")
    if res is not None:
        if res["pairs"]:
            items = "".join(f"<li>{a} &nbsp;↔&nbsp; {b}</li>" for a, b in res["pairs"])
            st.markdown(
                f'<div class="ds-card" style="border-left:5px solid var(--teal);padding:14px 18px;margin-bottom:8px;">'
                f'<strong style="color:var(--teal-deep);">🔗 Linked {len(res["pairs"])} pair(s)</strong>'
                f'<ul style="margin:8px 0 0 0;padding-left:18px;color:var(--ink);font-size:0.9rem;">{items}</ul></div>',
                unsafe_allow_html=True)
        else:
            st.info("No new relationships found — your notes don't have obvious connections yet.")

    # id → title lookup for rendering related links
    id2title = {c["metadata"].get("id"): c["metadata"].get("title", "") for c in concepts}

    col1, col2 = st.columns([1, 2])
    with col1:
        cat_filter = st.selectbox("Filter by category", ["All", "Todo", "Reminder", "Expense", "General Note"], key="vault_cat")
    with col2:
        search_query = st.text_input("Search notes / transcripts", placeholder="Type keywords...")

    filtered = []
    for c in concepts:
        meta, body = c["metadata"], c["body"]
        if cat_filter != "All" and meta.get("category", "") != cat_filter:
            continue
        if search_query:
            q = search_query.lower()
            if q not in meta.get("title", "").lower() and q not in body.lower():
                continue
        filtered.append(c)

    st.markdown(f'<p style="font-size:0.82rem;color:#94A3B8;margin:6px 0 14px 0;">'
                f'Showing <strong style="color:#0F172A;">{len(filtered)}</strong> of {len(concepts)} concepts.</p>',
                unsafe_allow_html=True)
    if not filtered:
        st.info("No concepts match your filter and search.")
        return

    for c in filtered:
        meta = c["metadata"]
        cat = meta.get("category", "General Note")
        title = meta.get("title", "Untitled Concept")
        ts = meta.get("timestamp", "")
        done = meta.get("done", False)
        rel_ids = meta.get("related") or []
        if not isinstance(rel_ids, list):
            rel_ids = [rel_ids]
        tag = "  ✓ done" if done else ""
        link_tag = f"  🔗 {len(rel_ids)}" if rel_ids else ""
        label = f"{CAT_ICON.get(cat,'📝')}  [{cat}]  {title} — {ts}{tag}{link_tag}"
        with st.expander(label):
            if rel_ids:
                chips = " · ".join(f"🔗 {id2title.get(r, str(r)[:8])}" for r in rel_ids)
                st.markdown(
                    f'<div style="background:#F0FDFA;border:1px solid #CCFBF1;border-radius:10px;'
                    f'padding:8px 12px;margin-bottom:12px;font-size:0.85rem;color:#0D9488;font-weight:600;">'
                    f'Related: {chips}</div>', unsafe_allow_html=True)
            m1, m2 = st.columns([2, 1])
            with m1:
                st.markdown("**📄 Markdown Content**")
                # Drop the OKF '## Related' block from the *display* — its /id.md links are
                # bundle-relative (valid in the file for portability, but a dead link in the app).
                # The "Related:" chip above already surfaces the connections.
                body_display = re.sub(r"\n*## Related\n(?:- .*\n?)*", "\n", c["body"]).rstrip()
                st.markdown(body_display)
            with m2:
                st.markdown("**⚙️ OKF Frontmatter**")
                try:
                    st.code(yaml.dump(meta, sort_keys=False, default_flow_style=False), language="yaml")
                except Exception:
                    st.code(str(meta), language="json")
                st.markdown(f"**File:** `knowledge/{c['filename']}`")
                if meta.get("audio_path") and os.path.exists(meta["audio_path"]):
                    st.markdown("**Original voice memo:**")
                    st.audio(meta["audio_path"])

                # Permanent delete (CSV row + OKF file + audio), with a confirm step
                st.markdown("---")
                fid = c["filename"].replace(".md", "")
                tid = meta.get("id", fid)
                confirm_key = f"delc_{fid}"
                if st.session_state.get(confirm_key):
                    st.markdown('<span style="color:#DC2626;font-weight:600;font-size:0.85rem;">Delete this note permanently?</span>',
                                unsafe_allow_html=True)
                    d1, d2 = st.columns(2)
                    with d1:
                        if st.button("Yes, delete", key=f"delyes_{fid}", type="primary", width="stretch"):
                            db_helper.delete_task(tid)
                            st.session_state.pop(confirm_key, None)
                            st.toast("Note deleted", icon="🗑️")
                            st.rerun()
                    with d2:
                        if st.button("Cancel", key=f"delno_{fid}", type="secondary", width="stretch"):
                            st.session_state.pop(confirm_key, None)
                            st.rerun()
                else:
                    if st.button("🗑️ Delete note", key=f"del_{fid}", type="secondary", width="stretch"):
                        st.session_state[confirm_key] = True
                        st.rerun()


# ═══════════════════════ PAGE: PRIVATE (confidential, passcode-gated) ═════════
def render_private():
    st.markdown('<p class="ds-eyebrow">Confidential</p>', unsafe_allow_html=True)
    st.markdown('<p class="ds-page-title">🔒 Private</p>', unsafe_allow_html=True)
    st.markdown('<p class="ds-help">Notes kept only on this device — <strong>never</strong> sent to the AI and '
                'never written to the OKF bundle. Protected by a passcode.</p>', unsafe_allow_html=True)

    items = db_helper.get_confidential_items()

    if not st.session_state.get("private_unlocked"):
        first = not db_helper.has_private_passcode()
        msg = "Set a passcode to protect your private notes." if first else "Enter your passcode to view."
        st.markdown(
            f'<div class="ds-card" style="max-width:440px;"><div style="font-size:1.8rem;">🔒</div>'
            f'<div style="font-weight:700;color:var(--ink);margin:6px 0 2px 0;">{len(items)} private note(s) · locked</div>'
            f'<div style="color:var(--muted);font-size:0.85rem;">{msg}</div></div>', unsafe_allow_html=True)
        pw = st.text_input("Passcode", type="password", key="priv_pw_input")
        if st.button("Set passcode & unlock" if first else "Unlock", type="primary", key="priv_unlock"):
            if not pw:
                st.warning("Enter a passcode.")
            elif db_helper.check_private_passcode(pw):
                st.session_state.private_unlocked = True
                st.rerun()
            else:
                st.error("Wrong passcode.")
        return

    top1, top2 = st.columns([3, 1], vertical_alignment="center")
    with top1:
        st.markdown(f'<p class="ds-help" style="margin:0;">Unlocked · {len(items)} private note(s).</p>', unsafe_allow_html=True)
    with top2:
        if st.button("🔒 Lock", key="priv_lock", width="stretch"):
            st.session_state.private_unlocked = False
            st.rerun()

    if not items:
        st.markdown('<div class="ds-empty" style="margin-top:10px;"><h4>No private notes yet</h4>'
                    '<p>Capture one via <strong>Capture → 🔒 Private</strong>.</p></div>', unsafe_allow_html=True)
        return

    st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)
    for it in items:
        due = _fmt_due(it.get("due_date", ""), it.get("due_time", ""))
        due_html = f'<span class="ds-time-chip">🕑 {due}</span>' if due else ""
        done = str(it.get("done", "")).lower() in ("true", "1", "yes")
        strike = "text-decoration:line-through;color:var(--muted);" if done else ""
        with st.container(key=f"prow-{it['id']}"):
            pc1, pc2 = st.columns([0.82, 0.18], vertical_alignment="center")
            with pc1:
                st.markdown(
                    f'<div class="ds-task"><div><span class="ds-badge badge-confidential">Confidential</span>'
                    f'<span class="ds-task-title" style="{strike}">{it.get("summary","")}</span></div>{due_html}</div>',
                    unsafe_allow_html=True)
            with pc2:
                if st.button("Delete", key=f"pdel-{it['id']}", width="stretch"):
                    db_helper.delete_task(it["id"])
                    st.rerun()
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)


# ═══════════════════════ NAVIGATION + TOP BAR ════════════════════════════════
stats = db_helper.get_stats()
inbox_count = stats["needs_detail"]

overview_page = st.Page(render_overview, title="Overview", icon=":material/dashboard:", url_path="overview", default=True)
capture_page = st.Page(render_capture, title="Capture", icon=":material/mic:", url_path="capture")
agenda_page = st.Page(render_agenda, title="Agenda", icon=":material/checklist:", url_path="agenda")
inbox_page = st.Page(render_inbox, title="Inbox", icon=":material/inbox:", url_path="inbox")
ask_page = st.Page(render_ask, title="Ask", icon=":material/forum:", url_path="ask")
vault_page = st.Page(render_vault, title="Vault", icon=":material/hub:", url_path="vault")
private_page = st.Page(render_private, title="Private", icon=":material/lock:", url_path="private")

nav = st.navigation([overview_page, capture_page, agenda_page, inbox_page, ask_page, vault_page, private_page], position="hidden")
current_title = getattr(nav, "title", "") or "Overview"

# Clear per-page transient state when the user switches tabs, so stale banners/cards
# (e.g. the Capture "Filed" result) don't linger after you navigate away and back.
_cur_path = getattr(nav, "url_path", "") or "overview"
if st.session_state.get("_prev_page") != _cur_path:
    st.session_state.last_captured_task = None
    st.session_state.pop("private_saved", None)
    st.session_state.pop("_pending_q", None)
    st.session_state.link_result = None
    st.session_state.confirm_demo = False
    # Re-lock private notes whenever you leave the Private page, so confidential
    # items are re-masked on the Agenda and everywhere else.
    if _cur_path != "private":
        st.session_state.private_unlocked = False
    st.session_state._prev_page = _cur_path

# Row 1 — centered brand (click goes home)
logo_img = f'<img src="{LOGO_URI}" alt="DaySync AI" />' if LOGO_URI else ""
st.markdown(f'<a href="/" target="_self" class="ds-brand-link">{logo_img}'
            f'<span class="ds-nav-name">DaySync AI</span></a>', unsafe_allow_html=True)

# Row 2 — scrollable tab bar + a settings gear
with st.container(key="ds-topnav"):
    tc = st.columns([1, 1, 1, 1, 1, 1, 1, 0.7], vertical_alignment="center")
    with tc[0]:
        st.page_link(overview_page, label="Overview", icon=":material/dashboard:")
    with tc[1]:
        st.page_link(capture_page, label="Capture", icon=":material/mic:")
    with tc[2]:
        st.page_link(agenda_page, label="Agenda", icon=":material/checklist:")
    with tc[3]:
        st.page_link(inbox_page, label=f"Inbox ({inbox_count})" if inbox_count else "Inbox", icon=":material/inbox:")
    with tc[4]:
        st.page_link(ask_page, label="Ask", icon=":material/forum:")
    with tc[5]:
        st.page_link(vault_page, label="Vault", icon=":material/hub:")
    with tc[6]:
        st.page_link(private_page, label="Private", icon=":material/lock:")
    with tc[7]:
        with st.container(key="ds-menu"):
            with st.popover("Settings", use_container_width=False):
                api_key_env = os.environ.get("GEMINI_API_KEY", "")
                api_key_input = st.text_input("Gemini API Key", value=st.session_state.get("api_key", api_key_env),
                                              type="password", help="Falls back to the GEMINI_API_KEY environment variable if left blank.")
                st.session_state.api_key = api_key_input if api_key_input else api_key_env
                if st.session_state.api_key:
                    st.markdown('<span class="ds-pill ds-pill-ok"><span class="ds-dot ds-dot-ok"></span>API key connected</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="ds-pill ds-pill-miss"><span class="ds-dot ds-dot-miss"></span>API key missing</span>', unsafe_allow_html=True)
                st.markdown("---")
                if st.session_state.get("confirm_demo"):
                    st.caption("Replace your notes with the demo bundle?")
                    dm1, dm2 = st.columns(2)
                    if dm1.button("Load", key="demo_yes", type="primary", width="stretch"):
                        db_helper.load_demo_data()
                        st.session_state.confirm_demo = False
                        st.toast("✨ Demo data loaded", icon="✨")
                        st.rerun()
                    if dm2.button("Cancel", key="demo_no", width="stretch"):
                        st.session_state.confirm_demo = False
                        st.rerun()
                else:
                    if st.button("✨ Load demo data", key="load_demo", width="stretch"):
                        st.session_state.confirm_demo = True
                        st.rerun()
                st.caption("OKF v0.1 compliant · google-genai SDK")

st.markdown('<hr class="ds-nav-sep" />', unsafe_allow_html=True)

# Floating quick actions — hidden on Ask (chat input owns the bottom) and Vault (delete
# controls live in the same corner)
if (getattr(nav, "url_path", "") or "") not in ("ask", "vault", "private"):
    with st.container(key="ds-fab"):
        st.page_link(ask_page, label="Ask", icon=":material/forum:")
        st.page_link(capture_page, label="Record", icon=":material/mic:")

nav.run()
