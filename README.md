# OpenAlex Author Utilities (Institutions + Citations)

A small Python toolkit built on top of **pyalex** to:
- resolve an author by **name + institution** (with optional fuzzy/LLM matching),
- infer an author’s **yearly main institution** from publication metadata,
- build **institution segments** and detect **migration events**,
- fetch **citation history** (author-level and publication-aggregated).

> Data source: OpenAlex (via `pyalex`).  
> Key idea for migrations: for each year, vote over institutions found in the author’s authorship block across the author’s works that year; pick the majority (tie-break prefers `type == "education"`), optionally smooth weak years, then segment and detect changes.

---

## Quick start

### 1) Install
```bash
pip install pyalex pandas
```

(Optional, only needed for timeline plots)
```bash
pip install plotly
```

### 2) Configure OpenAlex polite pool email
In your script (or in the module), set:
```python
import pyalex
pyalex.config.email = "your_email@example.com"
```

### 3) Run a simple example (migration history)
```python
author_id = "https://openalex.org/A5024763828"

hist = get_author_migration_history(
    author_id,
    min_votes_per_year=3,
    min_confidence=0.5,
    carry_forward_if_weak=True,
    min_streak_years=2,
)

print(hist["institution_segments"])
print(hist["migration_events"])
```

---

## What’s included

### Utility table

| Utility | Purpose | Input | Output | Example |
|---|---|---|---|---|
| `pick_main_institution(insts)` | Pick one “main” institution from a list (prefers `type=="education"`) | `insts: list[dict]` | `dict \| None` | `pick_main_institution(authorship["institutions"])` |
| `extract_target_authorship(work, author_id)` | Find the target author’s authorship block inside a Work | `work: dict`, `author_id: str` | `dict \| None` | `au = extract_target_authorship(w, "A5024763828")` |
| `select_best_institution(inst_counter, inst_by_id)` | Choose top-voted institution with deterministic tie-break | `Counter`, `dict[id->inst]` | `dict \| None` | `best = select_best_institution(counter, inst_by_id)` |
| `majority_institution_last_years(author_id, years_window=2, max_works_to_scan=500)` | Majority institution over the latest `years_window` publication years | `author_id: str` | `(inst: dict\|None, latest_year: int\|None)` | `inst, y = majority_institution_last_years("A...")` |
| `latest_main_institution(author_id, fallback_years_window=2)` | “Latest institution”: try most recent work; fallback to majority of last years | `author_id: str` | `(inst\|None, reference_time, method)` | `inst, when, method = latest_main_institution("A...")` |
| `institution_matches(user_inst, found_inst)` | Direct matching against OpenAlex institution id/ror/name | `user_inst: str`, `found_inst: dict` | `bool` | `institution_matches("KAUST", inst)` |
| `institution_matches_vague(user_inst, found_inst)` | Vague matching (calls `same_university` if available) | `user_inst: str`, `found_inst: dict` | `bool` | `institution_matches_vague("UIUC", inst)` |
| `resolve_author_by_name_and_institution(author_name, institution_input, fallback_years_window=2, vague=True)` | Resolve an author record by name + institution constraint | `name: str`, `inst: str` | `dict \| None` | `resolve_author_by_name_and_institution("Bernard Ghanem","KAUST")` |
| `list_all_publications(author_id, per_page=200, n_max=None)` | Generator over all Works (oldest → newest) | `author_id: str` | iterator of Work dicts | `for w in list_all_publications("A..."):` |
| `get_author_citation_history(author_id)` | Author-level citation history from OpenAlex Author record | `author_id: str` | dict with `counts_by_year` | `hist = get_author_citation_history("A...")` |
| `get_author_citation_history_from_publications(author_id, max_works_to_scan=20000, per_page=200)` | Aggregate citations by year by iterating Works’ `counts_by_year` | `author_id: str` | dict with `counts_by_year_sum` + consistency stats | `get_author_citation_history_from_publications("A...")` |
| `build_yearly_main_institution(author_id, ..., min_votes_per_year=3, min_confidence=0.5, carry_forward_if_weak=True)` | Build year→main institution series using voting + smoothing | `author_id: str` | dict with `years: list[...]` | `yearly = build_yearly_main_institution("A...")` |
| `summarize_institution_segments(yearly)` | Merge consecutive years with same institution into segments | yearly dict | `list[segments]` | `segs = summarize_institution_segments(yearly)` |
| `detect_migration_events(segments, min_streak_years=2)` | Detect migrations between segments (new segment must last long enough) | segments list | list of migration events | `events = detect_migration_events(segs, 2)` |
| `get_author_migration_history(author_id, ...)` | One-call wrapper: yearly → segments → migrations | `author_id: str` | dict containing everything | `hist = get_author_migration_history("A...")` |

---

## Usage recipes

### 1) Resolve an author by name + institution
```python
result = resolve_author_by_name_and_institution(
    author_name="bernard ghanem",
    institution_input="KAUST",
    fallback_years_window=2,
    vague=True,
)

if result is None:
    print("No match found.")
else:
    print(result["author_id"], result["latest_main_institution"]["display_name"])
```

### 2) Build yearly main institution timeline
```python
yearly = build_yearly_main_institution(
    author_id="https://openalex.org/A5024763828",
    min_votes_per_year=3,
    min_confidence=0.5,
    carry_forward_if_weak=True,
)
print(yearly["years"][:5])
```

### 3) Get segments and migrations
```python
hist = get_author_migration_history(
    "https://openalex.org/A5024763828",
    min_votes_per_year=3,
    min_confidence=0.5,
    carry_forward_if_weak=True,
    min_streak_years=2,
)

for s in hist["institution_segments"]:
    print(f"{s['start_year']}-{s['end_year']}: {s['institution_name']}")

for e in hist["migration_events"]:
    print(f"{e['year']}: {e['from_institution']['name']} -> {e['to_institution']['name']}")
```

### 4) Citation history (author record)
```python
c = get_author_citation_history("https://openalex.org/A5024763828")
print(c["counts_by_year"][-10:])
```

### 5) Citation history (publication aggregation)
```python
c2 = get_author_citation_history_from_publications("https://openalex.org/A5024763828")
print(c2["counts_by_year_sum"][-10:])
print("Unassigned citations estimate:", c2["unassigned_citations_estimate"])
```

---

## Parameters and behavior

### Voting logic
- Each Work can contribute **multiple votes** if the author lists multiple institutions on that Work.
- Majority is selected per year; ties are broken deterministically:
  1) prefer institutions where `type == "education"`,
  2) else pick smallest institution id lexicographically.

### Smoothing logic
- A year is **reliable** if:
  - `total_votes >= min_votes_per_year` AND
  - `confidence >= min_confidence` where `confidence = top_votes / total_votes`.
- If `carry_forward_if_weak=True`, an unreliable year inherits the last reliable institution.

### Migration detection
- After segmenting consecutive years, a transition is a migration only if the **new segment length** is at least `min_streak_years`.

---

## Notes / limitations
- OpenAlex affiliations can be missing or noisy for some Works. Majority voting + confidence gating helps, but cannot guarantee correctness.
- Some Works may have no `institutions` listed in the authorship block; those Works are skipped for voting.
- `institution_matches_vague` optionally calls `same_university()` (not included here). If you don’t provide it, matching falls back to direct id/ror/name heuristics.

---

## License
Add your preferred license (MIT/Apache-2.0/etc.) in `LICENSE`.
