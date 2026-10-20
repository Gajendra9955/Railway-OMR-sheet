"""Marks calculation. Refuses to produce a score from unreliable inputs."""


def _num(x, default):
    try:
        return float(x)
    except Exception:
        return default


def calculate(key, omr, settings=None):
    s = settings or {}
    correct_m = _num(s.get("correct"), 1.0)
    neg_m = abs(_num(s.get("negative"), 1.0 / 3.0))
    multiple_policy = s.get("multiple", "zero")          # zero | wrong
    dropped_policy = s.get("dropped", "full")            # full | normal

    problems = []
    if not key or key.get("blocking"):
        problems.append(dict(level="error", code="key_unreliable", message="उत्तर कुंजी भरोसे से नहीं पढ़ी गई, इसलिए अंक नहीं निकाले गए।"))
    if not omr or omr.get("blocking"):
        problems.append(dict(level="error", code="omr_unreliable", message="OMR शीट भरोसे से नहीं पढ़ी गई, इसलिए अंक नहीं निकाले गए।"))
    if problems:
        return dict(ok=False, problems=problems)

    answers = key["answers"]
    responses = omr["responses"]
    n_omr = len(responses)
    n_key = key["max_q"]
    if n_key != n_omr:
        return dict(ok=False, problems=[dict(
            level="error", code="count_mismatch",
            message=f"उत्तर कुंजी में {n_key} प्रश्न हैं पर OMR शीट में {n_omr} प्रश्न पहचाने गए। दोनों फ़ाइलें एक ही परीक्षा की होनी चाहिए; अंक नहीं निकाले गए।")])

    rows, tot = [], 0.0
    cnt = dict(correct=0, wrong=0, unattempted=0, multiple=0, unclear=0, bonus=0)
    for r in responses:
        q = r["q"]
        k = answers.get(str(q))
        act = (key.get("actions", {}).get(str(q), "") or "").lower()
        is_bonus = dropped_policy == "full" and ("drop" in act or "bonus" in act)
        marks, res = 0.0, None
        if is_bonus:
            res, marks = "bonus", correct_m
        elif r["status"] == "answered":
            if r["answer"] == k:
                res, marks = "correct", correct_m
            else:
                res, marks = "wrong", -neg_m
        elif r["status"] == "blank":
            res = "unattempted"
        elif r["status"] == "multiple":
            res = "multiple"
            marks = -neg_m if multiple_policy == "wrong" else 0.0
        else:  # faint / ambiguous
            res = "unclear"
        cnt[res] += 1
        tot += marks
        rows.append(dict(q=q, key=k, marked=r["answer"], status=r["status"], result=res, marks=round(marks, 4), action=act or None,
                         crop=r.get("crop")))
    total = round(tot, 2)
    return dict(
        ok=True, problems=[],
        total=total,
        min_possible=round(total - neg_m * cnt["unclear"], 2),
        max_possible=round(total + correct_m * cnt["unclear"], 2),
        counts=cnt, n=n_omr,
        attempted=cnt["correct"] + cnt["wrong"],
        accuracy=round(100.0 * cnt["correct"] / max(1, cnt["correct"] + cnt["wrong"]), 1),
        settings=dict(correct=correct_m, negative=round(neg_m, 4), multiple=multiple_policy, dropped=dropped_policy),
        rows=rows,
    )
