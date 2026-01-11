import pyalex
from pyalex import Authors, Works
from collections import Counter

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
