"""Answer-key reading.

Pipeline:
  1. words with positions: searchable-PDF text layer, or OCR (scanned PDF / image)
  2. group words into table rows by vertical position
  3. read records  QNO [QID] ANSWER [ACTION]  from each row (position-based, never by order)
  4. validate: duplicates, missing numbers, unreadable / ambiguous answers
"""
import io
import re
from collections import defaultdict
from statistics import median

import cv2
import numpy as np

from .fileio import FileProblem, load_pages, sniff

QNO_RE = re.compile(r"^\d{1,3}$")
QID_RE = re.compile(r"^\d{6,}$")
COMPACT_RE = re.compile(r"^(\d{1,3})\s*[-.:)]\s*\(?([A-Da-d])\)?$")
LETTER_RE = re.compile(r"^[A-Da-d]$")
MULTI_RE = re.compile(r"^[A-Da-d]([,/&+\-][A-Da-d])+$")
STRIP = " |¦[]{}()<>,;:_~`'\"."
LOW_CONF = 55
MAX_Q = 500


# ----------------------------------------------------------------------------
# 1. words
# ----------------------------------------------------------------------------
def words_from_pdf_text(data: bytes):
    import pdfplumber

    words = []
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for pi, page in enumerate(pdf.pages):
                for w in page.extract_words(x_tolerance=2, y_tolerance=2):
                    words.append(
                        dict(text=w["text"], x0=w["x0"], x1=w["x1"], top=w["top"], bottom=w["bottom"], page=pi + 1, conf=None)
                    )
    except Exception:
        return []
    return words


def _estimate_skew(gray):
    """Projection-profile skew estimate in degrees (small angles)."""
    h, w = gray.shape
    s = 900.0 / max(h, w)
    small = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else gray
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    best, best_a = -1.0, 0.0
    hh, ww = bw.shape
    c = (ww / 2, hh / 2)
    for a in np.arange(-4.0, 4.01, 0.5):
        M = cv2.getRotationMatrix2D(c, a, 1.0)
        r = cv2.warpAffine(bw, M, (ww, hh), flags=cv2.INTER_NEAREST)
        v = np.var(r.sum(axis=1).astype(np.float64))
        if v > best:
            best, best_a = v, a
    return best_a


def _remove_table_lines(gray):
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = gray.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 25), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, h // 40)))
    lines = cv2.bitwise_or(cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk), cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk))
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    out = gray.copy()
    out[lines > 0] = 255
    return out


def prepare_for_ocr(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w < 1700:
        f = 1700.0 / w
        gray = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    ang = _estimate_skew(gray)
    if abs(ang) >= 0.25:
        hh, ww = gray.shape
        M = cv2.getRotationMatrix2D((ww / 2, hh / 2), ang, 1.0)
        gray = cv2.warpAffine(gray, M, (ww, hh), flags=cv2.INTER_CUBIC, borderValue=255)
    return _remove_table_lines(gray)


def words_from_image(img_bgr, page_no=1, psm=6, prepared=None):
    import pytesseract

    gray = prepare_for_ocr(img_bgr) if prepared is None else prepared
    d = pytesseract.image_to_data(gray, lang="eng", config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    words = []
    for i, t in enumerate(d["text"]):
        t = (t or "").strip()
        try:
            conf = float(d["conf"][i])
        except Exception:
            conf = -1
        if not t or conf < 0:
            continue
        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        words.append(dict(text=t, x0=x, x1=x + w, top=y, bottom=y + h, page=page_no, conf=conf))
    return words


# ----------------------------------------------------------------------------
# 2/3. rows and records
# ----------------------------------------------------------------------------
def group_lines(words):
    lines = []
    pages = defaultdict(list)
    for w in words:
        pages[w["page"]].append(w)
    for page in sorted(pages):
        ws = sorted(pages[page], key=lambda w: (w["top"] + w["bottom"]) / 2)
        hs = [w["bottom"] - w["top"] for w in ws if w["bottom"] > w["top"]]
        tol = 0.6 * (median(hs) if hs else 10)
        cur, cur_y = [], None
        for w in ws:
            yc = (w["top"] + w["bottom"]) / 2
            if cur and abs(yc - cur_y) <= tol:
                cur.append(w)
                cur_y = sum((c["top"] + c["bottom"]) / 2 for c in cur) / len(cur)
            else:
                if cur:
                    lines.append(sorted(cur, key=lambda w: w["x0"]))
                cur, cur_y = [w], yc
        if cur:
            lines.append(sorted(cur, key=lambda w: w["x0"]))
    return lines


def _clean(t):
    return t.strip(STRIP)


def _answer_of(tok, numeric_ok):
    """-> (answer or None, issue or None). Lower-case letters only count when a question id is present too."""
    t = _clean(tok)
    if LETTER_RE.match(t) and (numeric_ok or t.isupper()):
        return t.upper(), None
    if MULTI_RE.match(t):
        return None, "multi_answer"
    if numeric_ok and re.fullmatch(r"[1-4]", t):
        return "ABCD"[int(t) - 1], None
    return None, "unreadable"


def parse_lines(lines):
    records = []
    for line in lines:
        toks = [(_clean(w["text"]), w) for w in line]
        toks = [(t, w) for t, w in toks if t]
        n = len(toks)
        i = 0
        while i < n:
            t, w = toks[i]
            m = COMPACT_RE.match(t)
            if m:
                records.append(
                    dict(q=int(m.group(1)), qid=None, ans=m.group(2).upper(), issue=None, action="", raw=t,
                         page=w["page"], conf=w["conf"], y=w["top"])
                )
                i += 1
                continue
            if QID_RE.match(t) and i + 2 < n and QNO_RE.match(toks[i + 1][0]):
                # layout: QID  QNO  ANSWER
                a, issue = _answer_of(toks[i + 2][0], True)
                if a or issue == "multi_answer":
                    q = int(toks[i + 1][0])
                    j = i + 3
                    action, j = _collect_action(toks, j)
                    records.append(_rec(q, t, a, issue, action, toks[i + 2], w, toks[i + 1][1]))
                    i = j
                    continue
            if QNO_RE.match(t) and 1 <= int(t) <= MAX_Q:
                j = i + 1
                qid = None
                if j < n and QID_RE.match(toks[j][0]):
                    qid = toks[j][0]
                    j += 1
                if j < n:
                    a, issue = _answer_of(toks[j][0], numeric_ok=qid is not None)
                    if a or issue == "multi_answer":
                        action, j2 = _collect_action(toks, j + 1)
                        records.append(_rec(int(t), qid, a, issue, action, toks[j], w, w))
                        i = j2
                        continue
                    if qid is not None:
                        # qno + id present but the answer cell is not readable
                        records.append(
                            dict(q=int(t), qid=qid, ans=None, issue="unreadable", action="", raw=toks[j][0],
                                 page=w["page"], conf=toks[j][1]["conf"], y=w["top"], bbox=_bbox(toks[j][1]))
                        )
                        i = j
                        continue
                elif qid is not None:
                    records.append(
                        dict(q=int(t), qid=qid, ans=None, issue="unreadable", action="", raw="",
                             page=w["page"], conf=w["conf"], y=w["top"])
                    )
                    i = j
                    continue
            i += 1
    return records


def _collect_action(toks, j):
    parts = []
    while j < len(toks) and len(parts) < 3:
        t = toks[j][0]
        if re.fullmatch(r"[A-Za-z][A-Za-z\-]{2,}", t):
            parts.append(t)
            j += 1
        else:
            break
    return " ".join(parts), j


def _bbox(w):
    return (w["x0"], w["top"], w["x1"], w["bottom"])


def _rec(q, qid, ans, issue, action, ans_tok, qno_w, qid_w):
    confs = [c for c in (ans_tok[1]["conf"], qno_w["conf"]) if c is not None]
    return dict(q=q, qid=qid, ans=ans, issue=issue, action=action, raw=ans_tok[0], page=qno_w["page"],
                conf=min(confs) if confs else None, y=qno_w["top"], bbox=_bbox(ans_tok[1]))


def recover_cells(records, prepared):
    """Re-read doubtful answer cells (unreadable / low confidence) one by one, restricted to A-D."""
    import pytesseract

    for r in records:
        bad = (not r["ans"] and r.get("issue") == "unreadable") or (r["ans"] and r["conf"] is not None and r["conf"] < LOW_CONF)
        if not bad or not r.get("bbox") or r["page"] not in prepared:
            continue
        g = prepared[r["page"]]
        x0, y0, x1, y1 = r["bbox"]
        pad = 10
        crop = g[max(0, int(y0) - pad):int(y1) + pad, max(0, int(x0) - pad):int(x1) + pad]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        crop = cv2.copyMakeBorder(crop, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        reads = []
        for psm in (10, 7, 8):
            # tesseract reports confidence 0 in whitelist mode, so agreement between modes is the evidence instead
            t = pytesseract.image_to_string(crop, lang="eng", config=f"--psm {psm} -c tessedit_char_whitelist=ABCD").strip()
            reads.append(t)
        votes = [t for t in reads if len(t) == 1 and t in "ABCD"]
        if len(votes) >= 2 and len(set(votes)) == 1 and len([t for t in reads if t and t != votes[0]]) == 0:
            r["ans"], r["issue"], r["conf"] = votes[0], None, 85.0


def _read_digits(g, bbox):
    """Re-read one cell as digits only; accept only if three tesseract modes agree."""
    import pytesseract

    x0, y0, x1, y1 = [int(v) for v in bbox]
    crop = g[max(0, y0 - 8):y1 + 8, max(0, x0 - 12):x1 + 12]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    crop = cv2.copyMakeBorder(crop, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    reads = []
    for psm in (7, 8, 10):
        t = pytesseract.image_to_string(crop, lang="eng", config=f"--psm {psm} -c tessedit_char_whitelist=0123456789").strip()
        reads.append(t)
    if reads[0] and reads[0].isdigit() and len(set(reads)) == 1:
        return int(reads[0])
    return None


def fill_row_gaps(recs, prepared):
    """OCR sometimes drops a whole table row. Look for vertical gaps in each answer column and re-read just that band."""
    have = {r["q"] for r in recs}
    added = []
    by_page = defaultdict(list)
    for r in recs:
        if r.get("bbox"):
            by_page[r["page"]].append(r)
    for pi, rs in by_page.items():
        g = prepared.get(pi)
        if g is None or len(rs) < 4:
            continue
        cols = []
        for r in sorted(rs, key=lambda r: r["bbox"][0]):
            if cols and abs(cols[-1][0]["bbox"][0] - r["bbox"][0]) < 60:
                cols[-1].append(r)
            else:
                cols.append([r])
        for col in cols:
            col.sort(key=lambda r: r["bbox"][1])
            if len(col) < 4:
                continue
            dys = [b["bbox"][1] - a["bbox"][1] for a, b in zip(col[:-1], col[1:])]
            pitch = median(dys)
            for a, b, dy in zip(col[:-1], col[1:], dys):
                if pitch > 5 and dy > 1.6 * pitch:
                    y0 = int(a["bbox"][3] + 0.25 * pitch)
                    y1 = int(b["bbox"][1] - 0.25 * pitch)
                    if y1 - y0 < 10:
                        continue
                    band = g[max(0, y0):min(g.shape[0], y1), :]
                    band = cv2.copyMakeBorder(band, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)
                    for psm in (6, 7):
                        got = parse_lines(group_lines(words_from_image(None, pi, psm, prepared=band)))
                        new = [x for x in got if x["q"] not in have and min(a["q"], b["q"]) - 1 <= x["q"] <= max(a["q"], b["q"]) + 1 + int(dy / pitch)]
                        for x in new:
                            x["bbox"] = None            # coordinates belong to the band, not the page
                            have.add(x["q"])
                            added.append(x)
                        if new:
                            break
                    else:
                        # the row text is readable but its question-number cell is not: re-read that cell as digits
                        ws = sorted(words_from_image(None, pi, 6, prepared=band), key=lambda w: w["x0"])
                        for i, w in enumerate(ws):
                            if QID_RE.match(_clean(w["text"])) and 1 <= i and i + 1 < len(ws):
                                ans, _ = _answer_of(ws[i + 1]["text"], True)
                                if not ans or (ws[i + 1]["conf"] or 0) < LOW_CONF:
                                    continue
                                val = _read_digits(band, (ws[i - 1]["x0"], ws[i - 1]["top"], ws[i - 1]["x1"], ws[i - 1]["bottom"]))
                                if val is not None and a["q"] < val < b["q"] and val not in have:
                                    have.add(val)
                                    added.append(dict(q=val, qid=_clean(w["text"]), ans=ans, issue=None, action="", raw=ans,
                                                      page=pi, conf=ws[i + 1]["conf"], y=0, bbox=None))
    return recs + added


# ----------------------------------------------------------------------------
# 4. validation
# ----------------------------------------------------------------------------
def build_key(records, source):
    problems = []
    by_q = defaultdict(list)
    for r in records:
        by_q[r["q"]].append(r)

    answers, actions, qids = {}, {}, {}
    for q in sorted(by_q):
        rs = by_q[q]
        good = [r for r in rs if r["ans"]]
        bad = [r for r in rs if not r["ans"]]
        vals = sorted({r["ans"] for r in good})
        if len(vals) > 1:
            problems.append(dict(level="error", code="duplicate_conflict", q=q,
                                 message=f"प्रश्न {q} दो बार मिला और उत्तर अलग-अलग हैं ({', '.join(vals)})।"))
            continue
        if len(rs) > 1 and len(vals) == 1 and not bad:
            problems.append(dict(level="warning", code="duplicate_same", q=q,
                                 message=f"प्रश्न {q} कुंजी में एक से अधिक बार है (उत्तर एक ही है: {vals[0]})।"))
        if not vals:
            r = bad[0]
            if r["issue"] == "multi_answer":
                msg = f"प्रश्न {q}: एक से अधिक उत्तर लिखे दिख रहे हैं ('{r['raw']}')।"
            else:
                msg = f"प्रश्न {q}: सही उत्तर पढ़ा नहीं जा सका (पढ़ा गया: '{r['raw']}')।"
            problems.append(dict(level="error", code="unreadable", q=q, message=msg))
            continue
        r = good[0]
        if r["conf"] is not None and r["conf"] < LOW_CONF:
            problems.append(dict(level="error", code="low_confidence", q=q,
                                 message=f"प्रश्न {q}: उत्तर '{r['ans']}' साफ़ नहीं पढ़ा गया (भरोसा {int(r['conf'])}%)। बेहतर स्कैन/फ़ोटो दें।"))
            continue
        answers[q] = r["ans"]
        actions[q] = r["action"]
        qids[q] = r["qid"]

    if by_q:
        lo, hi = min(by_q), max(by_q)
        missing = [q for q in range(1, hi + 1) if q not in by_q]
        if missing:
            shown = ", ".join(map(str, missing[:30])) + (" ..." if len(missing) > 30 else "")
            problems.append(dict(level="error", code="missing", q=None,
                                 message=f"उत्तर कुंजी में {len(missing)} प्रश्नों के उत्तर नहीं मिले: {shown}"))
    else:
        problems.append(dict(level="error", code="empty", q=None,
                             message="उत्तर कुंजी में कोई (प्रश्न संख्या → सही विकल्प) पंक्ति नहीं मिली।"))

    blocking = any(p["level"] == "error" for p in problems)
    return dict(
        source=source,
        answers={str(k): v for k, v in answers.items()},
        actions={str(k): v for k, v in actions.items() if v},
        qids={str(k): v for k, v in qids.items() if v},
        count=len(answers),
        max_q=max(by_q) if by_q else 0,
        problems=problems,
        blocking=blocking,
    )


# ----------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------
def read_answer_key(data: bytes):
    kind = sniff(data)
    if kind is None:
        raise FileProblem("bad_format", "यह फ़ाइल PDF, JPG या PNG नहीं लगती।")

    if kind == "pdf":
        words = words_from_pdf_text(data)
        records = parse_lines(group_lines(words)) if len(words) >= 10 else []
        if len(records) >= 3:
            return build_key(records, "pdf-text")
        # scanned / image-only PDF -> OCR
    pages = load_pages(data, dpi=220)[1]
    prepared = {pi: prepare_for_ocr(img) for pi, img in enumerate(pages, 1)}
    best = None
    for psm in (6, 4):
        recs = []
        for pi in prepared:
            recs += parse_lines(group_lines(words_from_image(None, pi, psm, prepared=prepared[pi])))
        recs = fill_row_gaps(recs, prepared)
        recover_cells(recs, prepared)
        if best is None or _score(recs) > _score(best):
            best = recs
        if _looks_complete(recs):
            break
    if len(best) < 3:
        # maybe the page is rotated: try 90/180/270 and pick the best
        cand = []
        for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
            prep_r = {pi: prepare_for_ocr(cv2.rotate(img, rot)) for pi, img in enumerate(pages, 1)}
            recs = []
            for pi in prep_r:
                recs += parse_lines(group_lines(words_from_image(None, pi, 6, prepared=prep_r[pi])))
            recover_cells(recs, prep_r)
            cand.append(recs)
        top = max(cand, key=_score)
        if _score(top) > _score(best):
            best = top
    return build_key(best, "ocr")


def _score(recs):
    good = {r["q"] for r in recs if r["ans"] and (r["conf"] is None or r["conf"] >= LOW_CONF)}
    return len(good)


def _looks_complete(recs):
    qs = sorted({r["q"] for r in recs})
    return bool(qs) and qs[0] == 1 and qs[-1] - qs[0] + 1 == len(qs) and all(
        r["ans"] and (r["conf"] is None or r["conf"] >= LOW_CONF) for r in recs)
