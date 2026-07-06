# DaySync AI — Sample Data & OKF Bundle

This folder is a **committed showcase** so you can see what DaySync produces without running it.
The live app's own data (`../agent_vault.csv`, `../knowledge/`, `../media/`) is git-ignored because it
is regenerated at runtime — this `examples/` bundle is a curated snapshot instead.

## What's here

| Path | What it is |
|---|---|
| `agent_vault.csv` | Sample vault (the app's CSV database) with 9 varied notes. |
| `okf-bundle/` | The matching **Open Knowledge Format (OKF) v0.1** bundle the app generates. |
| `okf-bundle/index.md` | Bundle root — declares `okf_version` and links every concept. |
| `okf-bundle/log.md` | Change log (captured / completed / linked history). |
| `okf-bundle/<id>.md` | One conformant concept document per note. |

## Features this sample demonstrates

- **All four categories** — Todo, Reminder, Expense, General Note.
- **Due dates & times** — an **overdue** todo (Submit the expense report), one due **today**
  (Buy groceries), and items due **tomorrow** (Chennai trip).
- **Expenses with amounts** — Pay the electricity bill (1500), Bought groceries (2400).
- **Completed tasks** — Buy medicines is marked `done: true`.
- **Human-in-the-loop** — "Pay the bill" has `needs_review: true` (a missing detail to confirm).
- **Concept cross-links (knowledge graph)** — related notes link to each other via OKF's
  bundle-relative link convention (`[title](/id.md)`) plus a `related:` frontmatter list:
  - Buy groceries ↔ Bought groceries (task ↔ its expense)
  - Book tatkal ticket ↔ Pack bags (same trip)
  - Buy medicines ↔ "walk 30 min daily" (same health thread)

## Load it into the app

Open the app's **☰ Menu → Load demo data** to copy this bundle into your live vault and explore
it interactively. (Regenerate this folder with `python scripts/build_examples.py` if you change the sample.)
