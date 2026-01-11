
from pyalex import Authors


from typing import Dict, Any, List, Optional, Tuple
from collections import Counter, defaultdict

import pyalex
from pyalex import Works

pyalex.config.email = "peihaoli1014@example.com"

# If you put same_university() in another file, import it like:
# from ai_utils import same_university
# For this example, we assume same_university is available in scope.
try:
    from ai_utils import same_university
except Exception:
    same_university = None  # allows running without LLM, only exact matching


def pick_main_institution(insts):
    if not insts:
        return None
    if len(insts) == 1:
        return insts[0]

    edu = [i for i in insts if i.get("type") == "education"]
    if edu:
        return edu[0]

    return insts[0]


def extract_target_authorship(work, author_id):
    target_full = f"https://openalex.org/{author_id}"
    for au in (work.get("authorships") or []):
        au_id = (au.get("author") or {}).get("id") or ""
        if au_id == target_full or au_id.endswith(author_id):
            return au
    return None


def select_best_institution(inst_counter, inst_by_id):
    if not inst_counter:
        return None

    top_count = inst_counter.most_common(1)[0][1]
    top_ids = [iid for iid, c in inst_counter.items() if c == top_count]

    edu_ids = [
        iid for iid in top_ids if (inst_by_id.get(iid) or {}).get("type") == "education"
    ]
    chosen_id = sorted(edu_ids)[0] if edu_ids else sorted(top_ids)[0]

    return inst_by_id.get(chosen_id)


def majority_institution_last_years(author_id, years_window=2, max_works_to_scan=500):
    query = (
        Works()
        .filter(author={"id": author_id})
        .sort(publication_date="desc")
        .select(["id", "publication_year", "publication_date", "authorships"])
    )

    inst_counter = Counter()
    inst_by_id = {}
    latest_year = None

    for page in query.paginate(per_page=200, n_max=max_works_to_scan):
        for w in page:
            y = w.get("publication_year")
            if not y:
                continue

            if latest_year is None:
                latest_year = y

            if y < latest_year - (years_window - 1):
                best = select_best_institution(inst_counter, inst_by_id)
                return best, latest_year

            au = extract_target_authorship(w, author_id)
            if not au:
                continue

            insts = au.get("institutions") or []
            for inst in insts:
                inst_id = inst.get("id")
                if not inst_id:
                    continue
                inst_counter[inst_id] += 1
                inst_by_id[inst_id] = inst

    best = select_best_institution(inst_counter, inst_by_id)
    return best, latest_year


def latest_main_institution(author_id, fallback_years_window=2):
    works = (
        Works()
        .filter(author={"id": author_id})
        .sort(publication_date="desc")
        .select(["id", "publication_date", "publication_year", "authorships"])
        .get()
    )

    if not works:
        return None, None, "no_works"

    w = works[0]
    work_date = w.get("publication_date") or str(w.get("publication_year") or "")

    au = extract_target_authorship(w, author_id)
    if au:
        insts = au.get("institutions") or []
        inst = pick_main_institution(insts)
        if inst:
            return inst, work_date, "most_recent_work"

    inst2, latest_year = majority_institution_last_years(
        author_id, years_window=fallback_years_window
    )
    if inst2:
        if latest_year is not None and fallback_years_window == 2:
            ref = f"{latest_year} and {latest_year - 1}"
        elif latest_year is not None:
            ref = str(latest_year)
        else:
            ref = None
        return inst2, ref, "majority_last_years"

    return None, work_date, "none"


def normalize_text(s):
    return (s or "").strip().lower()


def institution_matches(user_inst, found_inst):
    u = normalize_text(user_inst)
    if not u:
        return False

    inst_id = (found_inst.get("id") or "")
    inst_ror = (found_inst.get("ror") or "")
    inst_name = normalize_text(found_inst.get("display_name") or "")

    u_no_prefix = u.replace("https://openalex.org/", "")

    if u == normalize_text(inst_id) or u_no_prefix == normalize_text(inst_id).replace("https://openalex.org/", ""):
        return True
    if u == normalize_text(inst_ror):
        return True

    return (u in inst_name) or (inst_name in u)


def institution_matches_vague(user_inst, found_inst):
    """
    Vague matching:
      - First try basic string matching (institution_matches)
      - If mismatch, call same_university(user_inst, found_name) to double-check.
    Returns True/False.
    """
    if institution_matches(user_inst, found_inst):
        return True

    if same_university is None:
        return False

    found_name = found_inst.get("display_name") or ""
    try:
        return same_university(user_inst, found_name) == 1
    except Exception:
        return False


def resolve_author_by_name_and_institution(
    author_name,
    institution_input,
    fallback_years_window=2,
    vague=True,
):
    """
    Input: author_name (str), institution_input (str)
    Output: dict with author info if match found, else None

    If vague=True (default):
      - if exact/substring match fails, call same_university(institution_input, latest_inst_name)
        to double-check.
    """
    cands = Authors().search(author_name).get()

    for cand in cands:
        author_id = (cand.get("id") or "").replace("https://openalex.org/", "")
        if not author_id:
            continue

        a = Authors()[author_id]

        inst, when, method = latest_main_institution(
            author_id, fallback_years_window=fallback_years_window
        )
        if not inst:
            continue

        matched = (
            institution_matches_vague(institution_input, inst)
            if vague
            else institution_matches(institution_input, inst)
        )

        if matched:
            return {
                "author_id": a.get("id"),
                "display_name": a.get("display_name"),
                "orcid": a.get("orcid"),
                "works_count": a.get("works_count"),
                "cited_by_count": a.get("cited_by_count"),
                "latest_main_institution": {
                    "id": inst.get("id"),
                    "display_name": inst.get("display_name"),
                    "ror": inst.get("ror"),
                    "country_code": inst.get("country_code"),
                    "type": inst.get("type"),
                },
                "reference_time": when,
                "institution_method": method,
                "matched_via": "vague_llm" if vague and not institution_matches(institution_input, inst) else "direct",
            }

    return None

# if __name__ == "__main__":
#     name = "bernard ghanem"
#     inst = "KAUST"  # vague user input
#
#     result = resolve_author_by_name_and_institution(
#         name,
#         inst,
#         fallback_years_window=2,
#         vague=True,  # default True
#     )
#
#     if result is None:
#         print("No match found.")
#     else:
#         print("Match found:")
#         print(result)


def _normalize_author_id(author_id: str) -> str:
    """
    Accepts:
      - "A5024763828"
      - "https://openalex.org/A5024763828"
    Returns:
      - "A5024763828"
    """
    if not isinstance(author_id, str):
        raise TypeError("author_id must be a string")
    s = author_id.strip()
    if not s:
        raise ValueError("author_id is empty")
    return s.replace("https://openalex.org/", "")


def list_all_publications(author_id: str, per_page: int = 200, n_max=None):
    """
    List ALL publications (OpenAlex Works objects) for the given author,
    ordered from oldest to newest (front to back).

    n_max:
      - None => fetch all
      - int  => fetch up to n_max works
    """
    aid = _normalize_author_id(author_id)

    query = (
        Works()
        .filter(author={"id": aid})
        .sort(publication_date="asc")  # from old -> new
    )

    # paginate returns pages (lists of works)
    for page in query.paginate(per_page=per_page, n_max=n_max):
        for w in page:
            yield w


def get_author_citation_history(author_id: str):
    """
    Fetch author-level citation history from OpenAlex.

    Input:
      author_id: either "A...." or full URL "https://openalex.org/A...."

    Output:
      dict with:
        - author_id (full OpenAlex URL)
        - display_name
        - orcid
        - works_count
        - cited_by_count (total)
        - counts_by_year (list of {"year": int, "cited_by_count": int}) sorted by year asc
    """
    if not isinstance(author_id, str) or not author_id.strip():
        raise ValueError("author_id must be a non-empty string")

    aid = author_id.strip()
    if aid.startswith("https://openalex.org/"):
        aid = aid.replace("https://openalex.org/", "")

    a = Authors()[aid]  # fetch full author record

    counts_by_year = a.get("counts_by_year") or []
    # normalize + sort
    norm = []
    for row in counts_by_year:
        y = row.get("year")
        c = row.get("cited_by_count")
        if isinstance(y, int) and isinstance(c, int):
            norm.append({"year": y, "cited_by_count": c})
    norm.sort(key=lambda x: x["year"])

    return {
        "author_id": a.get("id"),
        "display_name": a.get("display_name"),
        "orcid": a.get("orcid"),
        "works_count": a.get("works_count"),
        "cited_by_count": a.get("cited_by_count"),
        "counts_by_year": norm,
    }

def _normalize_author_id(author_id: str) -> str:
    """
    Accepts:
      - "A123..."
      - "https://openalex.org/A123..."
    Returns:
      - "A123..."
    """
    if not isinstance(author_id, str) or not author_id.strip():
        raise ValueError("author_id must be a non-empty string")

    aid = author_id.strip()
    if aid.startswith("https://openalex.org/"):
        aid = aid.replace("https://openalex.org/", "")
    return aid


def get_author_citation_history_from_publications(
    author_id: str,
    max_works_to_scan: int = 20000,
    per_page: int = 200,
):
    """
    Publication-oriented citation aggregation:
      - Sum cited_by_count across all works by the author
      - Aggregate counts_by_year across all works (when available)

    Notes / limitations:
      - Some Works do NOT include counts_by_year. Those citations will contribute to the total
        cited_by_count, but cannot be assigned to specific years. We'll report them as "unassigned".
      - For Works that have counts_by_year, the sum of their yearly counts may not always equal
        cited_by_count (data updates / indexing differences). We'll report a consistency check.

    Input:
      author_id: "A...." or "https://openalex.org/A...."

    Output dict:
      {
        "author_id": "https://openalex.org/A...." (normalized full URL),
        "works_scanned": int,
        "works_with_yearly_breakdown": int,
        "works_missing_yearly_breakdown": int,

        "cited_by_count_sum": int,              # sum of works' cited_by_count
        "counts_by_year_sum": [                # year-aggregated from works counts_by_year
           {"year": 2018, "cited_by_count": 123},
           ...
        ],

        "yearly_citations_sum": int,            # sum over counts_by_year_sum
        "unassigned_citations_estimate": int,   # cited_by_count_sum - yearly_citations_sum (>=0 in normal cases)
        "consistency": {
            "works_where_sum_yearly_gt_total": int,
            "works_where_sum_yearly_lt_total": int,
            "works_where_sum_yearly_eq_total": int,
        },
      }
    """
    aid = _normalize_author_id(author_id)
    full_author_url = f"https://openalex.org/{aid}"

    year_counter = Counter()

    works_scanned = 0
    works_with_yearly = 0
    works_missing_yearly = 0

    cited_by_count_sum = 0
    yearly_citations_sum_per_work_stats = {
        "works_where_sum_yearly_gt_total": 0,
        "works_where_sum_yearly_lt_total": 0,
        "works_where_sum_yearly_eq_total": 0,
    }

    query = (
        Works()
        .filter(author={"id": aid})
        .select(["id", "cited_by_count", "counts_by_year"])  # keep payload small
    )

    # Paginate through (potentially) all works
    for page in query.paginate(per_page=per_page, n_max=max_works_to_scan):
        for w in page:
            works_scanned += 1

            # 1) total citations (per work)
            c_total = w.get("cited_by_count")
            if isinstance(c_total, int) and c_total >= 0:
                cited_by_count_sum += c_total
            else:
                c_total = 0  # treat missing/invalid as 0

            # 2) yearly citations (per work, if present)
            cby = w.get("counts_by_year") or []
            if not cby:
                works_missing_yearly += 1
                continue

            works_with_yearly += 1
            sum_yearly_for_work = 0

            for row in cby:
                y = row.get("year")
                c = row.get("cited_by_count")
                if isinstance(y, int) and isinstance(c, int) and c >= 0:
                    year_counter[y] += c
                    sum_yearly_for_work += c

            # Consistency check vs cited_by_count
            if sum_yearly_for_work > c_total:
                yearly_citations_sum_per_work_stats["works_where_sum_yearly_gt_total"] += 1
            elif sum_yearly_for_work < c_total:
                yearly_citations_sum_per_work_stats["works_where_sum_yearly_lt_total"] += 1
            else:
                yearly_citations_sum_per_work_stats["works_where_sum_yearly_eq_total"] += 1

    # Build sorted history
    counts_by_year_sum = [
        {"year": y, "cited_by_count": year_counter[y]} for y in sorted(year_counter.keys())
    ]
    yearly_citations_sum = sum(year_counter.values())

    # "Unassigned" are citations from works that lack yearly breakdown (and/or mismatches)
    unassigned = cited_by_count_sum - yearly_citations_sum
    if unassigned < 0:
        # Data inconsistencies can lead to negative; clamp and still report.
        unassigned = 0

    return {
        "author_id": full_author_url,
        "works_scanned": works_scanned,
        "works_with_yearly_breakdown": works_with_yearly,
        "works_missing_yearly_breakdown": works_missing_yearly,
        "cited_by_count_sum": cited_by_count_sum,
        "counts_by_year_sum": counts_by_year_sum,
        "yearly_citations_sum": yearly_citations_sum,
        "unassigned_citations_estimate": unassigned,
        "consistency": yearly_citations_sum_per_work_stats,
    }

def _year_from_work(w: dict) -> Optional[int]:
    y = w.get("publication_year")
    if isinstance(y, int):
        return y
    # fallback from publication_date like "2020-03-04"
    d = w.get("publication_date")
    if isinstance(d, str) and len(d) >= 4 and d[:4].isdigit():
        return int(d[:4])
    return None


def _choose_year_main_institution(
    inst_counter: Counter,
    inst_by_id: Dict[str, dict],
) -> Tuple[Optional[dict], int, int, float]:
    """
    Returns:
      (chosen_inst_obj_or_None, top_votes, total_votes, confidence)
    """
    total = int(sum(inst_counter.values()))
    if total <= 0:
        return None, 0, 0, 0.0

    chosen = select_best_institution(inst_counter, inst_by_id)
    if not chosen:
        return None, 0, total, 0.0

    chosen_id = chosen.get("id")
    top = int(inst_counter.get(chosen_id, 0)) if chosen_id else 0
    conf = float(top) / float(total) if total else 0.0
    return chosen, top, total, conf


def build_yearly_main_institution(
    author_id: str,
    per_page: int = 200,
    max_works_to_scan: int = 20000,
    min_votes_per_year: int = 3,
    min_confidence: float = 0.5,
    carry_forward_if_weak: bool = True,
) -> Dict[str, Any]:
    """
    Publication-oriented institution timeline.

    For each year:
      - Count institutions from the author's authorship block across all works in that year
      - Pick the majority institution (tie-break prefers education)
      - Compute confidence = top_votes / total_votes

    Artifact handling:
      - Wrong-paper noise tends to be sparse; yearly majority voting suppresses it.
      - You can require min_votes_per_year and min_confidence for a year to be "reliable".
      - If carry_forward_if_weak=True, weak years inherit the previous reliable institution.

    Output:
      {
        "author_id": "https://openalex.org/A....",
        "years": [
          {
            "year": 2018,
            "main_institution": {...} or None,
            "main_inst_id": str or None,
            "main_inst_name": str or None,
            "top_votes": int,
            "total_votes": int,
            "confidence": float,
            "is_reliable": bool,
          },
          ...
        ]
      }
    """
    aid = _normalize_author_id(author_id)
    full_author_url = f"https://openalex.org/{aid}"

    # year -> Counter(inst_id -> votes)
    year_inst_counter: Dict[int, Counter] = defaultdict(Counter)
    # year -> inst_by_id map
    year_inst_by_id: Dict[int, Dict[str, dict]] = defaultdict(dict)

    works_scanned = 0

    query = (
        Works()
        .filter(author={"id": aid})
        .select(["id", "publication_year", "publication_date", "authorships"])
        .paginate(per_page=per_page, n_max=max_works_to_scan)
    )

    for page in query:
        for w in page:
            works_scanned += 1
            y = _year_from_work(w)
            if y is None:
                continue

            au = extract_target_authorship(w, aid)
            if not au:
                continue

            insts = au.get("institutions") or []
            # If the author has no institutions for this work, skip (common in some records)
            if not insts:
                continue

            # Vote for ALL institutions listed for that author on that work
            # (you can change to only pick_main_institution(insts) if you want single vote per paper)
            for inst in insts:
                inst_id = inst.get("id")
                if not inst_id:
                    continue
                year_inst_counter[y][inst_id] += 1
                year_inst_by_id[y][inst_id] = inst

    years_sorted = sorted(year_inst_counter.keys())
    out_years: List[Dict[str, Any]] = []

    last_reliable_inst: Optional[dict] = None

    for y in years_sorted:
        counter = year_inst_counter[y]
        inst_by_id = year_inst_by_id[y]

        chosen, top, total, conf = _choose_year_main_institution(counter, inst_by_id)

        is_reliable = (total >= min_votes_per_year) and (conf >= min_confidence)

        if (not is_reliable) and carry_forward_if_weak and last_reliable_inst is not None:
            # inherit previous stable institution (helps smooth sparse/noisy years)
            chosen = last_reliable_inst
        elif is_reliable and chosen is not None:
            last_reliable_inst = chosen

        out_years.append(
            {
                "year": y,
                "main_institution": chosen,
                "main_inst_id": (chosen or {}).get("id") if chosen else None,
                "main_inst_name": (chosen or {}).get("display_name") if chosen else None,
                "top_votes": top,
                "total_votes": total,
                "confidence": conf,
                "is_reliable": is_reliable,
            }
        )

    return {
        "author_id": full_author_url,
        "works_scanned": works_scanned,
        "years": out_years,
    }


def summarize_institution_segments(
    yearly: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Turn year-by-year main institution into continuous segments:
      KAUST: 2012-2019
      MIT:   2020-2024
      ...

    Returns list of:
      {
        "institution_id": str|None,
        "institution_name": str|None,
        "start_year": int,
        "end_year": int,
        "years": [int,...]
      }
    """
    rows = yearly.get("years") or []
    if not rows:
        return []

    segs: List[Dict[str, Any]] = []

    cur_id = rows[0].get("main_inst_id")
    cur_name = rows[0].get("main_inst_name")
    start = rows[0]["year"]
    years_bucket = [rows[0]["year"]]

    for r in rows[1:]:
        iid = r.get("main_inst_id")
        nm = r.get("main_inst_name")
        y = r["year"]

        if iid == cur_id:
            years_bucket.append(y)
            continue

        segs.append(
            {
                "institution_id": cur_id,
                "institution_name": cur_name,
                "start_year": start,
                "end_year": years_bucket[-1],
                "years": years_bucket[:],
            }
        )

        cur_id = iid
        cur_name = nm
        start = y
        years_bucket = [y]

    segs.append(
        {
            "institution_id": cur_id,
            "institution_name": cur_name,
            "start_year": start,
            "end_year": years_bucket[-1],
            "years": years_bucket[:],
        }
    )

    return segs


def detect_migration_events(
    segments: List[Dict[str, Any]],
    min_streak_years: int = 2,
) -> List[Dict[str, Any]]:
    """
    Migration heuristic:
      - Only treat a change as a "migration" if the new institution segment lasts >= min_streak_years.

    Returns list of events:
      {
        "from_institution": (id,name),
        "to_institution": (id,name),
        "year": int   # start year of new institution segment
      }
    """
    events: List[Dict[str, Any]] = []
    if not segments or len(segments) < 2:
        return events

    for prev, cur in zip(segments[:-1], segments[1:]):
        cur_len = (cur["end_year"] - cur["start_year"] + 1)
        if cur_len < min_streak_years:
            continue

        events.append(
            {
                "from_institution": {
                    "id": prev.get("institution_id"),
                    "name": prev.get("institution_name"),
                },
                "to_institution": {
                    "id": cur.get("institution_id"),
                    "name": cur.get("institution_name"),
                },
                "year": cur["start_year"],
            }
        )

    return events


def get_author_migration_history(
    author_id: str,
    per_page: int = 200,
    max_works_to_scan: int = 20000,
    min_votes_per_year: int = 3,
    min_confidence: float = 0.5,
    carry_forward_if_weak: bool = True,
    min_streak_years: int = 2,
) -> Dict[str, Any]:
    """
    One-call wrapper:
      - builds yearly main institution
      - segments it
      - detects migrations
    """
    yearly = build_yearly_main_institution(
        author_id=author_id,
        per_page=per_page,
        max_works_to_scan=max_works_to_scan,
        min_votes_per_year=min_votes_per_year,
        min_confidence=min_confidence,
        carry_forward_if_weak=carry_forward_if_weak,
    )
    segs = summarize_institution_segments(yearly)
    migrations = detect_migration_events(segs, min_streak_years=min_streak_years)

    return {
        "author_id": yearly["author_id"],
        "works_scanned": yearly["works_scanned"],
        "yearly_main_institution": yearly["years"],
        "institution_segments": segs,
        "migration_events": migrations,
        "params": {
            "min_votes_per_year": min_votes_per_year,
            "min_confidence": min_confidence,
            "carry_forward_if_weak": carry_forward_if_weak,
            "min_streak_years": min_streak_years,
        },
    }


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    author_id = "https://openalex.org/A5024763828"  # Bernard Ghanem
    hist = get_author_migration_history(
        author_id,
        min_votes_per_year=3,
        min_confidence=0.5,
        carry_forward_if_weak=True,
        min_streak_years=2,
    )

    print("Author:", hist["author_id"])
    print("Works scanned:", hist["works_scanned"])
    print("\nInstitution segments:")
    for s in hist["institution_segments"]:
        print(
            f"- {s['start_year']}-{s['end_year']}: {s['institution_name']} ({s['institution_id']})"
        )

    print("\nDetected migration events:")
    if not hist["migration_events"]:
        print("- (none found with current thresholds)")
    else:
        for e in hist["migration_events"]:
            print(
                f"- {e['year']}: {e['from_institution']['name']} -> {e['to_institution']['name']}"
            )