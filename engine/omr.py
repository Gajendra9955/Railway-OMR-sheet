"""OMR engine (OpenCV).

Pipeline
  1. render / load page
  2. find sheet boundary, fix perspective, normalise size
  3. detect bubble candidates (contours: size, aspect, circularity) and learn the dominant bubble size
  4. discover the grid from the bubbles themselves (columns -> blocks of 4 options -> rows); no fixed coordinates
  5. fix residual skew, re-detect
  6. verify question numbering by reading the printed numbers (also fixes 180-degree rotation)
  7. measure darkness inside every bubble relative to local paper + sheet baseline
  8. classify each question: answered / blank / multiple / faint / ambiguous
  9. compare detected layout with the supported template(s)
"""
import base64

import cv2
import numpy as np

from .fileio import load_pages

NORM_LONG = 2400
N_OPT = 4
OPTIONS = "ABCD"
FILL_T = 0.55      # contrast at/above which a bubble counts as filled
FAINT_T = 0.10     # between FAINT_T and FILL_T = faint / erased mark (shown for review, not scored)
LIGHT_T = 0.70     # a lone filled bubble below this is accepted but shown for review
CLOSE_T = 0.35     # a runner-up this dark next to a filled bubble makes the question ambiguous
MAX_PAGES_TRIED = 6

TEMPLATES = [
    dict(id="rrb-sm-110", name="Railway Station Master OMR (110 प्रश्न, 2 कॉलम)", blocks=2, options=4, total=110),
]


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def resize_long(img, long_side=NORM_LONG):
    h, w = img.shape[:2]
    s = long_side / float(max(h, w))
    if abs(s - 1) < 0.02:
        return img
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=interp)


def rotate_image(img, angle, border=255):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(border, border, border))


def rotate90s(img, rot):
    if rot == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rot == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


# ----------------------------------------------------------------------------
# 2. sheet boundary + perspective
# ----------------------------------------------------------------------------
def normalize_sheet(img):
    """Return (sheet_bgr, info). Finds the paper, corrects perspective, normalises size."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 1000.0 / max(h, w)
    small = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    info = dict(warped=False)
    quad = None
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        sh, sw = small.shape
        if cv2.contourArea(c) >= 0.25 * sh * sw:
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            for eps in (0.01, 0.015, 0.02, 0.03, 0.04, 0.06):
                ap = cv2.approxPolyDP(hull, eps * peri, True)
                if len(ap) == 4:
                    quad = ap.reshape(4, 2).astype(np.float32)
                    break
            if quad is None:
                quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    out = img
    if quad is not None:
        quad = order_points(quad / s)
        tl, tr, br, bl = quad
        W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        poly_area = cv2.contourArea(quad.reshape(-1, 1, 2))
        covers_all = poly_area >= 0.985 * h * w
        long_, short_ = max(W, H), max(1, min(W, H))
        plausible = 1.15 <= long_ / short_ <= 2.1 and short_ > 300
        if plausible and not covers_all:
            dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(quad, dst)
            out = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            info["warped"] = True
    return resize_long(out), info


# ----------------------------------------------------------------------------
# 3. bubble candidates
# ----------------------------------------------------------------------------
def cluster_1d(vals, tol):
    vals = np.asarray(vals, dtype=np.float64)
    order = np.argsort(vals)
    groups, cur = [], [order[0]]
    for idx in order[1:]:
        if vals[idx] - vals[cur[-1]] <= tol:
            cur.append(idx)
        else:
            groups.append(np.array(cur))
            cur = [idx]
    groups.append(np.array(cur))
    return groups


def find_bubbles(gray):
    """-> (array Nx3 of cx, cy, diameter), median diameter) for round-ish blobs of the dominant size."""
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bs = int(round(w * 0.035)) | 1
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, bs, 12)
    cnts, hier = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    dmin, dmax = 0.007 * w, 0.045 * w
    cand = []
    if hier is None:
        return np.zeros((0, 3)), 0.0
    for i, c in enumerate(cnts):
        if hier[0][i][3] != -1:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if not (dmin <= bw <= dmax and dmin <= bh <= dmax):
            continue
        ar = bw / float(bh)
        if not 0.7 <= ar <= 1.43:
            continue
        area = cv2.contourArea(c)
        per = cv2.arcLength(c, True)
        if per <= 0:
            continue
        circ = 4 * np.pi * area / (per * per)
        ext = area / float(bw * bh)
        if circ < 0.62 or ext < 0.6:
            continue
        cand.append((x + bw / 2.0, y + bh / 2.0, (bw + bh) / 2.0))
    if len(cand) < 8:
        return np.zeros((0, 3)), 0.0
    cand = np.array(cand)
    ds = cand[:, 2]
    best_n, best_d = 0, 0
    for d0 in ds:
        n = int(np.sum((ds >= 0.8 * d0) & (ds <= 1.2 * d0)))
        if n > best_n:
            best_n, best_d = n, d0
    keep = cand[(ds >= 0.8 * best_d) & (ds <= 1.2 * best_d)]
    return keep, float(np.median(keep[:, 2]))


def estimate_skew(b, d):
    angles = []
    for i in range(len(b)):
        dx = b[:, 0] - b[i, 0]
        dy = b[:, 1] - b[i, 1]
        m = np.where((dx > 1.2 * d) & (dx < 3.6 * d) & (np.abs(dy) < 0.6 * d))[0]
        if m.size:
            j = m[np.argmin(dx[m])]
            angles.append(np.degrees(np.arctan2(dy[j], dx[j])))
    if len(angles) < 20:
        return 0.0
    return float(np.median(angles))


# ----------------------------------------------------------------------------
# 4. grid discovery
# ----------------------------------------------------------------------------
def _fail(reason, message, **kw):
    return dict(ok=False, reason=reason, message=message, **kw)


def discover_grid(b, d, n_opt=N_OPT):
    if len(b) < 40:
        return _fail("few_bubbles", f"OMR में केवल {len(b)} बुलबुले मिले। फ़ोटो/स्कैन साफ़ और पूरा शीट दिखता हुआ होना चाहिए।")
    xs, ys = b[:, 0], b[:, 1]
    cols = []
    for g in cluster_1d(xs, 0.45 * d):
        cols.append(dict(x=float(xs[g].mean()), n=len(g), idx=g))
    mx = max(c["n"] for c in cols)
    cols = sorted([c for c in cols if c["n"] >= 0.55 * mx], key=lambda c: c["x"])
    if len(cols) == 0 or len(cols) % n_opt != 0:
        return _fail("columns", f"उत्तर-ग्रिड में {len(cols)} बुलबुला-कॉलम मिले; यह {n_opt} के गुणज में होना चाहिए (हर प्रश्न में {n_opt} विकल्प)।",
                     n_columns=len(cols))
    groups = [cols[i:i + n_opt] for i in range(0, len(cols), n_opt)]
    inner_gaps = []
    for g in groups:
        gx = np.diff([c["x"] for c in g])
        if gx.min() <= 0 or gx.max() / gx.min() > 1.4:
            return _fail("irregular", "विकल्पों (A,B,C,D) के बीच की दूरी अनियमित है; शीट मुड़ी या धुंधली हो सकती है।")
        inner_gaps += list(gx)
    inner = float(np.median(inner_gaps))
    for a, c in zip(groups[:-1], groups[1:]):
        if c[0]["x"] - a[-1]["x"] < 1.25 * inner:
            return _fail("blocks", "प्रश्न-कॉलम आपस में अलग नहीं पहचाने जा सके।")
    blocks = []
    for g in groups:
        centers = np.array([c["x"] for c in g])
        # every bubble near one of this block's option columns
        dist = np.abs(xs[:, None] - centers[None, :])
        near = np.where(dist.min(axis=1) <= 0.6 * d)[0]
        opt_of = dist.argmin(axis=1)
        rows = []
        for r in cluster_1d(ys[near], 0.5 * d):
            members = near[r]
            if len(set(opt_of[members].tolist())) >= 2:
                rows.append(float(ys[members].mean()))
        rows = sorted(rows)
        if len(rows) < 3:
            return _fail("rows", "एक प्रश्न-कॉलम में पंक्तियाँ पर्याप्त नहीं मिलीं।")
        rows = _core_rows(rows)
        if len(rows) < 3:
            return _fail("rows", "प्रश्न-पंक्तियों की दूरी नियमित नहीं मिली (शीट मुड़ी/धुंधली हो सकती है)।")
        rows = _fill_missing_rows(rows)
        # refine each option column's x from the bubbles that sit on the selected rows
        cx_ref = []
        for j, c in enumerate(g):
            sel = [xs[i] for i in near if opt_of[i] == j and np.min(np.abs(np.array(rows) - ys[i])) <= 0.5 * d]
            cx_ref.append(float(np.median(sel)) if len(sel) >= 3 else c["x"])
        cells = []
        for ry in rows:
            row = []
            for j, cxj in enumerate(cx_ref):
                cx, cy, inferred = cxj, ry, True
                dd = np.hypot(b[:, 0] - cxj, b[:, 1] - ry)
                k = int(np.argmin(dd))
                if dd[k] <= 0.6 * d:
                    cx, cy, inferred = float(b[k, 0]), float(b[k, 1]), False
                row.append((cx, cy, inferred))
            cells.append(row)
        blocks.append(dict(cols=cx_ref, rows=rows, cells=cells))
    return dict(ok=True, blocks=blocks, d=d, pitch=inner)


def _core_rows(rows):
    """Keep only rows that belong to the regular question grid (drops e.g. roll-number rows with a different pitch)."""
    rows = np.asarray(rows, dtype=np.float64)
    dy = np.diff(rows)
    if len(dy) < 2:
        return rows.tolist()
    best, p = -1, float(np.median(dy))
    for v in dy:
        n = int(np.sum(np.abs(dy - v) <= 0.08 * v))
        if n > best:
            best, p = n, float(np.mean(dy[np.abs(dy - v) <= 0.08 * v]))
    ok_gap = np.abs(dy - p) <= 0.15 * p
    core = np.zeros(len(rows), bool)
    core[:-1] |= ok_gap
    core[1:] |= ok_gap
    # largest chain of core rows whose neighbours are at most ~2 pitches apart (allows group gaps / one missing row)
    best_chain, cur = [], []
    for i in range(len(rows)):
        if not core[i]:
            if len(cur) > len(best_chain):
                best_chain = cur
            cur = []
            continue
        if cur and rows[i] - cur[-1] > 2.2 * p:
            if len(cur) > len(best_chain):
                best_chain = cur
            cur = []
        cur.append(rows[i])
    if len(cur) > len(best_chain):
        best_chain = cur
    return [float(v) for v in best_chain]


def _fill_missing_rows(rows):
    dy = np.diff(rows)
    pitch = float(np.median(dy))
    out = [rows[0]]
    for a, b_ in zip(rows[:-1], rows[1:]):
        k = (b_ - a) / pitch
        kr = int(round(k))
        if kr >= 2 and abs(k - kr) < 0.15:
            for t in range(1, kr):
                out.append(a + (b_ - a) * t / kr)
        out.append(b_)
    return out


# ----------------------------------------------------------------------------
# 6. numbering verification (reads the printed question numbers)
# ----------------------------------------------------------------------------
def _read_number(gray, x0, x1, y0, y1):
    import pytesseract

    H, W = gray.shape
    x0, x1, y0, y1 = int(max(0, x0)), int(min(W, x1)), int(max(0, y0)), int(min(H, y1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = gray[y0:y1, x0:x1]
    _, ink = cv2.threshold(cv2.GaussianBlur(crop, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink[:2, :] = 0
    ink[-2:, :] = 0
    ys, xs = np.where(ink > 0)
    if len(xs) < 12:
        return None
    # tight box around the ink, then rescale to a size tesseract likes
    bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
    if (by1 - by0) < 5 or (bx1 - bx0) > 0.9 * (x1 - x0):
        return None
    pad = 3
    tight = crop[max(0, by0 - pad):by1 + pad + 1, max(0, bx0 - pad):bx1 + pad + 1]
    sc = 44.0 / max(1, tight.shape[0])
    tight = cv2.resize(tight, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(tight, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.copyMakeBorder(bw, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)
    for psm in (7, 8):
        try:
            txt = pytesseract.image_to_string(bw, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789").strip()
        except Exception:
            return None
        digits = "".join(ch for ch in txt if ch.isdigit())
        if digits and len(digits) <= 3:
            return int(digits)
    return None


def _verify_pass(gray, layout, idx_fn):
    checked = readable = matches = 0
    q0 = 1
    d, pitch = layout["d"], layout["pitch"]
    prev_right = -1e9
    for blk in layout["blocks"]:
        n = len(blk["rows"])
        ax = blk["cols"][0]
        x1 = ax - 1.2 * (d / 2)
        x0 = max(ax - 2.4 * pitch, prev_right + 0.8 * d)
        for i in idx_fn(n):
            cy = blk["rows"][i]
            val = _read_number(gray, x0, x1, cy - 0.65 * d, cy + 0.65 * d)
            checked += 1
            if val is not None:
                readable += 1
                if val == q0 + i:
                    matches += 1
        q0 += n
        prev_right = blk["cols"][-1]
    return checked, readable, matches


def _is_verified(readable, matches):
    # a wrong numbering hypothesis almost never matches several printed numbers, so even a modest hit-rate is strong evidence
    ratio = matches / max(1, readable)
    return (readable >= 3 and ratio >= 0.6) or (matches >= 6 and ratio >= 0.3)


def verify_numbers(gray, layout):
    """Read printed question numbers at sampled rows; compare with the order we assume (block by block, top to bottom)."""
    def few(n):
        return sorted(set(np.linspace(0, n - 1, min(n, 6)).round().astype(int).tolist()))

    def more(n):
        return [i for i in range(1, n, 3) if i not in few(n)]

    checked, readable, matches = _verify_pass(gray, layout, few)
    if not _is_verified(readable, matches):
        c2, r2, m2 = _verify_pass(gray, layout, more)       # weak evidence: look at many more rows
        checked, readable, matches = checked + c2, readable + r2, matches + m2
    ratio = matches / max(1, readable)
    return dict(checked=checked, readable=readable, matches=matches,
                ok=_is_verified(readable, matches),
                bad=readable >= 5 and ratio <= 0.2)


# ----------------------------------------------------------------------------
# 7/8. measuring + classification
# ----------------------------------------------------------------------------
def cell_fraction(gray, cx, cy, r):
    H, W = gray.shape
    ri = max(2.0, 0.55 * r)
    R = int(np.ceil(1.9 * r))
    x0, x1 = max(0, int(round(cx)) - R), min(W, int(round(cx)) + R + 1)
    y0, y1 = max(0, int(round(cy)) - R), min(H, int(round(cy)) + R + 1)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0
    paper = float(np.percentile(patch, 92))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= ri * ri
    vals = patch[mask[: patch.shape[0], : patch.shape[1]]]
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals < 0.72 * paper))


def measure(gray, layout):
    r = layout["d"] / 2.0
    fr = []
    for blk in layout["blocks"]:
        fb = [[cell_fraction(gray, cx, cy, r) for (cx, cy, _) in row] for row in blk["cells"]]
        fr.append(np.array(fb))
    allv = np.concatenate([f.ravel() for f in fr])
    base = float(min(np.median(allv), 0.35))
    return [np.clip((f - base) / (1 - base), 0, 1) for f in fr], base


def classify(c):
    order = np.argsort(c)[::-1]
    top, second = float(c[order[0]]), float(c[order[1]])
    n_fill = int(np.sum(c >= FILL_T))
    flags = []
    if n_fill >= 2:
        return "multiple", None, flags
    if n_fill == 1:
        if second >= CLOSE_T:
            return "ambiguous", None, ["close_runner_up"]
        if second >= FAINT_T:
            flags.append("faint_other")
        if top < LIGHT_T:
            flags.append("light_mark")
        return "answered", OPTIONS[int(order[0])], flags
    if top >= FAINT_T:
        return "faint", None, flags
    return "blank", None, flags


# ----------------------------------------------------------------------------
# rendering helpers
# ----------------------------------------------------------------------------
def _jpeg_b64(img, q=70):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def row_crop(sheet_bgr, blk, i, d):
    x0 = int(max(0, blk["cols"][0] - 4.2 * d))
    x1 = int(min(sheet_bgr.shape[1], blk["cols"][-1] + 1.6 * d))
    cy = blk["rows"][i]
    y0, y1 = int(max(0, cy - 0.95 * d)), int(min(sheet_bgr.shape[0], cy + 0.95 * d))
    crop = sheet_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    s = 420.0 / crop.shape[1]
    crop = cv2.resize(crop, (420, max(10, int(crop.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return _jpeg_b64(crop, 75)


def overlay(sheet_bgr, layout, results):
    vis = sheet_bgr.copy()
    d = layout["d"]
    col = {"answered": (60, 170, 60), "blank": (170, 170, 170), "multiple": (0, 0, 220),
           "faint": (0, 140, 255), "ambiguous": (0, 140, 255)}
    for blk, res in zip(layout["blocks"], results):
        for i, row in enumerate(blk["cells"]):
            st = res[i]["status"]
            for j, (cx, cy, inferred) in enumerate(row):
                chosen = res[i]["answer"] == OPTIONS[j]
                cv2.circle(vis, (int(cx), int(cy)), int(d / 2 + 3), col[st] if (chosen or st in ("multiple", "faint", "ambiguous")) else (200, 200, 200), 3 if chosen else 1)
        for tag, i in (("first", 0), ("last", len(blk["rows"]) - 1)):
            cv2.rectangle(vis, (int(blk["cols"][0] - d), int(blk["rows"][i] - d * 0.8)),
                          (int(blk["cols"][-1] + d), int(blk["rows"][i] + d * 0.8)), (255, 120, 0), 2)
    s = 900.0 / vis.shape[1]
    vis = cv2.resize(vis, (900, int(vis.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return _jpeg_b64(vis, 70)


# ----------------------------------------------------------------------------
# page analysis
# ----------------------------------------------------------------------------
def detect_with_deskew(sheet_bgr):
    """-> (sheet_bgr (maybe rotated), gray, grid dict, skew_deg)"""
    gray = cv2.cvtColor(sheet_bgr, cv2.COLOR_BGR2GRAY)
    b, d = find_bubbles(gray)
    skew = 0.0
    if len(b) >= 40:
        a = estimate_skew(b, d)
        if 0.15 <= abs(a) <= 8:
            sheet_bgr = rotate_image(sheet_bgr, a)
            gray = cv2.cvtColor(sheet_bgr, cv2.COLOR_BGR2GRAY)
            b, d = find_bubbles(gray)
            skew = a
    if len(b) < 40:
        return sheet_bgr, gray, _fail("few_bubbles", f"OMR में केवल {len(b)} बुलबुले मिले। फ़ोटो/स्कैन साफ़ और पूरा शीट दिखता हुआ होना चाहिए।"), skew, len(b)
    return sheet_bgr, gray, discover_grid(b, d), skew, len(b)


def analyze_oriented(sheet):
    sheet, gray, grid, skew, nb = detect_with_deskew(sheet)
    res = dict(sheet=sheet, gray=gray, grid=grid, skew=skew, n_bubbles=nb, verify=None)
    if grid["ok"]:
        res["verify"] = verify_numbers(gray, grid)
    return res


def _rank(res):
    if not res["grid"]["ok"]:
        return (0, 0, res["n_bubbles"])
    v = res["verify"]
    return (1, v["matches"] - 0.5 * (v["readable"] - v["matches"]), 0)


def analyze_page(img):
    sheet, info = normalize_sheet(img)
    h, w = sheet.shape[:2]
    portrait = h >= w
    tried = []
    if portrait:
        r0 = analyze_oriented(sheet)
        r0["rot"] = 0
        tried.append(r0)
        if r0["grid"]["ok"] and (r0["verify"]["bad"] or r0["verify"]["readable"] < 3):
            r1 = analyze_oriented(rotate90s(sheet, 180))
            r1["rot"] = 180
            tried.append(r1)
    else:
        for rot in (90, 270):
            r = analyze_oriented(rotate90s(sheet, rot))
            r["rot"] = rot
            tried.append(r)
    best = max(tried, key=_rank)
    best["page_warped"] = info["warped"]
    return best


def build_result(best, force_layout=False):
    problems = []
    grid = best["grid"]
    if not grid["ok"]:
        problems.append(dict(level="error", code=grid["reason"], message=grid["message"]))
        return dict(ok=False, blocking=True, needs_confirmation=False, problems=problems, responses=[], layout=None,
                    template=None, counts={}, overlay=None)

    gray, sheet = best["gray"], best["sheet"]
    layout = grid
    fr, base = measure(gray, layout)
    responses, per_block = [], []
    q = 1
    d = layout["d"]
    for bi, (blk, f) in enumerate(zip(layout["blocks"], fr)):
        blk_res = []
        for i in range(len(blk["rows"])):
            st, ans, flags = classify(f[i])
            rec = dict(q=q, status=st, answer=ans, flags=flags, contrast=[round(float(x), 2) for x in f[i]])
            if any(cell[2] for cell in blk["cells"][i]):
                rec["flags"] = flags + ["grid_inferred"]
            if st in ("multiple", "faint", "ambiguous") or "faint_other" in flags or "light_mark" in flags:
                rec["crop"] = row_crop(sheet, blk, i, d)
            blk_res.append(rec)
            responses.append(rec)
            q += 1
        per_block.append(blk_res)
    total = len(responses)

    tpl = None
    for t in TEMPLATES:
        if len(layout["blocks"]) == t["blocks"] and total == t["total"]:
            tpl = t
            break
    rows_per_block = [len(b["rows"]) for b in layout["blocks"]]
    detected = f"{len(layout['blocks'])} कॉलम, हर कॉलम में {rows_per_block} पंक्तियाँ, कुल {total} प्रश्न"
    needs_conf = False
    if tpl is None:
        known = "; ".join(f"{t['name']}: {t['blocks']} कॉलम, कुल {t['total']} प्रश्न" for t in TEMPLATES)
        if force_layout:
            problems.append(dict(level="warning", code="template_forced",
                                 message=f"शीट समर्थित टेम्पलेट से मेल नहीं खाती; आपकी पुष्टि पर पहचाना गया लेआउट ({detected}) उपयोग किया गया।"))
        else:
            needs_conf = True
            problems.append(dict(level="error", code="template_mismatch",
                                 message=f"यह OMR शीट समर्थित टेम्पलेट से मेल नहीं खाती। पहचाना गया: {detected}। समर्थित: {known}। "
                                         "नीचे का चित्र देखकर लेआउट सही हो तो पुष्टि करें।"))

    v = best["verify"]
    if v["bad"]:
        problems.append(dict(level="error", code="numbering_mismatch",
                             message=f"शीट पर छपे प्रश्न-क्रमांक पहचाने गए क्रम से मेल नहीं खाते ({v['matches']}/{v['readable']} मिले)। "
                                     "शीट उल्टी/तिरछी हो सकती है या प्रश्नों का क्रम अलग है; इसलिए अंक नहीं निकाले गए।"))
    elif not v["ok"]:
        problems.append(dict(level="warning", code="numbering_unverified",
                             message=f"छपे हुए प्रश्न-क्रमांक OCR से पूरी तरह सत्यापित नहीं हो सके ({v['matches']}/{v['readable']} पढ़े गए क्रमांक मेल खाए)। "
                                     "प्रश्न-क्रम ऊपर-से-नीचे, बाएँ कॉलम के बाद दाएँ कॉलम माना गया है; चित्र से जाँच लें।"))

    counts = {k: sum(1 for r in responses if r["status"] == k) for k in ("answered", "blank", "multiple", "faint", "ambiguous")}
    if counts["ambiguous"] or counts["faint"]:
        problems.append(dict(level="warning", code="unclear_marks",
                             message=f"{counts['faint'] + counts['ambiguous']} प्रश्नों में हल्का/अस्पष्ट निशान मिला; इन्हें अंक-गणना में नहीं गिना गया (नीचे चित्र देखें)।"))
    blocking = any(p["level"] == "error" for p in problems)
    return dict(
        ok=not blocking, blocking=blocking, needs_confirmation=needs_conf, problems=problems,
        responses=responses, counts=counts, total=total,
        template=dict(matched=bool(tpl), id=tpl["id"] if tpl else None, name=tpl["name"] if tpl else None, detected=detected),
        layout=dict(blocks=len(layout["blocks"]), rows=rows_per_block, options=N_OPT),
        numbering=dict(readable=v["readable"], matches=v["matches"]),
        skew_deg=round(best["skew"], 2), rotated=best["rot"], baseline=round(base, 3),
        overlay=overlay(sheet, layout, per_block),
    )


def read_omr(data: bytes, force_layout=False):
    kind, pages = load_pages(data, dpi=200)
    best = None
    for img in pages[:MAX_PAGES_TRIED]:
        res = analyze_page(img)
        if best is None or _rank(res) > _rank(best):
            best = res
        if res["grid"]["ok"]:
            break
    out = build_result(best, force_layout)
    out["source"] = kind
    return out
