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
    """
    Detect printed OMR bubbles using OpenCV.
    Designed for Railway OMR sheets with pale pink outlines.
    Returns:
        bubbles: Nx3 array (center_x, center_y, diameter)
        median_diameter: median bubble diameter
    """
    h, w = gray.shape

    if h < 100 or w < 100:
        return np.zeros((0, 3), dtype=np.float32), 0.0

    # Normalize image size for more consistent bubble detection.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold: retain light printed bubble outlines.
    block_size = max(31, min(int(round(w * 0.025)) | 1, 151))

    th = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        5
    )

    # Join small gaps in bubble outlines without joining neighbouring bubbles.
    kernel_size = max(2, int(round(w * 0.0008)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(
        th,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None or not contours:
        return np.zeros((0, 3), dtype=np.float32), 0.0

    # Expected bubble size relative to normalized image width.
    dmin = max(6, 0.006 * w)
    dmax = 0.035 * w

    candidates = []

    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)

        if not (dmin <= bw <= dmax and dmin <= bh <= dmax):
            continue

        aspect = bw / float(bh)

        if not 0.75 <= aspect <= 1.33:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter <= 0:
            continue

        area = cv2.contourArea(contour)
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)

        # Reject lines and rectangular marks.
        if circularity < 0.40:
            continue

        # Approximate bubble center.
        moments = cv2.moments(contour)

        if moments["m00"] != 0:
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
        else:
            cx = x + bw / 2.0
            cy = y + bh / 2.0

        diameter = (bw + bh) / 2.0

        candidates.append((cx, cy, diameter))

    if len(candidates) < 8:
        return np.zeros((0, 3), dtype=np.float32), 0.0

    candidates = np.asarray(candidates, dtype=np.float32)

    # Estimate the dominant bubble diameter.
    diameters = candidates[:, 2]
    best_count = 0
    best_diameter = float(np.median(diameters))

    for diameter in diameters:
        similar = np.abs(diameters - diameter) <= 0.18 * diameter
        count = int(np.sum(similar))

        if count > best_count:
            best_count = count
            best_diameter = float(diameter)

    # Retain bubbles near the dominant diameter.
    keep_mask = (
        (diameters >= 0.82 * best_diameter) &
        (diameters <= 1.18 * best_diameter)
    )

    bubbles = candidates[keep_mask]

    if len(bubbles) < 8:
        return np.zeros((0, 3), dtype=np.float32), 0.0

    return bubbles, float(np.median(bubbles[:, 2]))


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
    """
    Detect OMR option columns and question rows.
    Supports two-block sheets such as Q1-55 on the left
    and Q56-110 on the right, with A/B/C/D options.
    """
    if len(b) < 40:
        return _fail(
            "few_bubbles",
            f"OMR में केवल {len(b)} बुलबुले मिले। "
            "कृपया साफ़ और पूरा OMR स्कैन अपलोड करें।"
        )

    xs = b[:, 0].astype(float)
    ys = b[:, 1].astype(float)

    # Step 1: Detect raw X-coordinate clusters.
    raw_cols = []
    for g in cluster_1d(xs, 0.45 * d):
        if len(g) == 0:
            continue

        raw_cols.append({
            "x": float(xs[g].mean()),
            "n": len(g),
            "idx": g
        })

    raw_cols.sort(key=lambda c: c["x"])

    if len(raw_cols) < n_opt:
        return _fail(
            "columns",
            f"केवल {len(raw_cols)} कॉलम पहचाने गए। "
            "स्कैन की स्पष्टता और बुलबुला पहचान जाँचें।",
            n_columns=len(raw_cols)
        )

    # Step 2: Remove only very weak/noisy clusters.
    # The previous 35% threshold could discard a valid option column.
    max_n = max(c["n"] for c in raw_cols)
    cols = [
        c for c in raw_cols
        if c["n"] >= max(3, 0.08 * max_n)
    ]
    cols.sort(key=lambda c: c["x"])

    # If filtering removed too many columns, retain raw clusters.
    if len(cols) < n_opt or len(cols) % n_opt != 0:
        cols = raw_cols

    if len(cols) % n_opt != 0:
        return _fail(
            "columns",
            f"{len(cols)} कॉलम पहचाने गए, लेकिन विकल्पों की "
            f"संख्या {n_opt} के गुणज में नहीं है। "
            "बेहतर रिज़ॉल्यूशन वाली OMR शीट अपलोड करें।",
            n_columns=len(cols)
        )

    # Step 3: Group option columns into question blocks.
    groups = [
        cols[i:i + n_opt]
        for i in range(0, len(cols), n_opt)
    ]

    inner_gaps = []

    for g in groups:
        gx = np.diff([c["x"] for c in g])

        if len(gx) != n_opt - 1 or np.any(gx <= 0):
            return _fail(
                "irregular",
                "A, B, C, D विकल्पों की स्थिति स्पष्ट नहीं है।"
            )

        # Allow more variation than the previous strict threshold.
        if gx.max() / gx.min() > 1.8:
            return _fail(
                "irregular",
                "विकल्पों के बीच दूरी अनियमित है। "
                "सीधी और साफ़ OMR शीट अपलोड करें।"
            )

        inner_gaps.extend(gx.tolist())

    inner = float(np.median(inner_gaps))

    # Step 4: Validate separation between blocks.
    for left, right in zip(groups[:-1], groups[1:]):
        gap = right[0]["x"] - left[-1]["x"]

        if gap < 1.10 * inner:
            return _fail(
                "blocks",
                "दो प्रश्न-ब्लॉक अलग नहीं पहचाने जा सके।"
            )

    # Step 5: Detect rows and bubble positions in every block.
    blocks = []

    for g in groups:
        centers = np.array([c["x"] for c in g], dtype=float)

        dist = np.abs(xs[:, None] - centers[None, :])
        opt_of = dist.argmin(axis=1)

        # Keep detected bubbles near one of the four option columns.
        near = np.where(dist.min(axis=1) <= 0.70 * d)[0]

        if len(near) < 12:
            return _fail(
                "rows",
                "प्रश्न-पंक्तियों के लिए पर्याप्त बुलबुले नहीं मिले।"
            )

        # Find rows using the detected bubble Y coordinates.
        row_groups = cluster_1d(ys[near], 0.55 * d)
        rows = []

        for r in row_groups:
            members = near[r]

            # A valid row should have bubbles in at least two options.
            distinct_options = len(set(opt_of[members].tolist()))

            if distinct_options >= 2:
                rows.append(float(np.median(ys[members])))

        rows = sorted(rows)

def discover_grid(b, d, n_opt=N_OPT):
    """
    Robust grid detection for Railway OMR sheets.
    Detects option columns using their X coordinates and
    validates the four-option pattern before building rows.
    """
    if b is None or len(b) < 40:
        return _fail(
            "few_bubbles",
            f"OMR में केवल {0 if b is None else len(b)} बुलबुले मिले। "
            "कृपया साफ़ और पूरा OMR स्कैन अपलोड करें।"
        )

    b = np.asarray(b, dtype=np.float64)
    xs = b[:, 0]
    ys = b[:, 1]

    # 1. Find approximate X-coordinate clusters.
    raw = []
    for idx in cluster_1d(xs, max(2.0, 0.45 * d)):
        if len(idx) == 0:
            continue
        raw.append({
            "x": float(np.median(xs[idx])),
            "n": len(idx),
            "idx": idx
        })

    raw.sort(key=lambda z: z["x"])

    if len(raw) < n_opt:
        return _fail(
            "columns",
            f"केवल {len(raw)} संभावित कॉलम मिले।",
            n_columns=len(raw)
        )

    # 2. Merge nearby duplicate detections, often caused by contours.
    merged = []
    merge_tol = max(2.0, 0.55 * d)

    for col in raw:
        if merged and col["x"] - merged[-1]["x"] < merge_tol:
            old = merged[-1]
            total = old["n"] + col["n"]
            old["x"] = (
                old["x"] * old["n"] + col["x"] * col["n"]
            ) / total
            old["n"] = total
        else:
            merged.append(dict(col))

    raw = merged

    # 3. Remove very weak clusters, but preserve plausible columns.
    max_n = max(c["n"] for c in raw)
    cols = [
        c for c in raw
        if c["n"] >= max(3, 0.06 * max_n)
    ]

    if len(cols) < n_opt:
        cols = raw

    cols.sort(key=lambda z: z["x"])

    # 4. Find four-column groups by regular spacing.
    # Do not assume every four consecutive clusters form a block.
    candidates = []

    for start in range(len(cols) - n_opt + 1):
        group = cols[start:start + n_opt]
        gx = np.diff([c["x"] for c in group])

        if np.any(gx <= 0):
            continue

        med_gap = float(np.median(gx))
        if med_gap <= 0:
            continue

        # Four option columns should have broadly similar spacing.
        ratio = float(gx.max() / gx.min())
        if ratio <= 1.8:
            score = ratio + 0.15 * abs(
                float(np.std(gx)) / med_gap
            )
            candidates.append((score, start, group, med_gap))

    if not candidates:
        return _fail(
            "irregular",
            "A, B, C, D विकल्पों के कॉलम स्पष्ट नहीं पहचाने गए। "
            "स्कैन या bubble detection में समस्या हो सकती है।"
        )

    # Select non-overlapping four-column groups.
    candidates.sort(key=lambda z: z[0])
    selected = []

    for score, start, group, gap in candidates:
        indices = set(range(start, start + n_opt))
        if any(indices.intersection(s[0]) for s in selected):
            continue
        selected.append((indices, group, gap))

    selected.sort(key=lambda z: z[1][0]["x"])

    # For this template, expect two blocks of four options.
    if len(selected) != 2:
        return _fail(
            "columns",
            f"OMR में {len(selected)} संभावित चार-विकल्प ब्लॉक मिले। "
            "इस शीट के लिए बाएँ और दाएँ दो ब्लॉक अपेक्षित हैं।",
            n_blocks=len(selected),
            n_columns=len(cols)
        )

    groups = [item[1] for item in selected]
    inner_gaps = []

    for group in groups:
        gx = np.diff([c["x"] for c in group])
        inner_gaps.extend(gx.tolist())

    inner = float(np.median(inner_gaps))

    if groups[1][0]["x"] - groups[0][-1]["x"] <= inner:
        return _fail(
            "blocks",
            "बाएँ और दाएँ प्रश्न-ब्लॉक अलग नहीं पहचाने जा सके।"
        )

    # 5. Build rows separately for each block.
    blocks = []

    for group in groups:
        centers = np.array([c["x"] for c in group], dtype=float)
        dist = np.abs(xs[:, None] - centers[None, :])
        opt_of = dist.argmin(axis=1)
        near = np.where(dist.min(axis=1) <= 0.70 * d)[0]

        if len(near) < 12:
            return _fail(
                "rows",
                "प्रश्न-पंक्तियों के लिए पर्याप्त बुलबुले नहीं मिले।"
            )

        row_groups = cluster_1d(ys[near], 0.55 * d)
        rows = []

        for rg in row_groups:
            members = near[rg]
            if len(set(opt_of[members].tolist())) >= 2:
                rows.append(float(np.median(ys[members])))

        rows = _core_rows(sorted(rows))

        if len(rows) < 3:
            return _fail(
                "rows",
                "प्रश्न-पंक्तियाँ नियमित रूप से नहीं मिलीं।"
            )

        rows = _fill_missing_rows(rows)

        # Refine each option's X position.
        cx_ref = []

        for j, col in enumerate(group):
            selected_x = [
                xs[i] for i in near
                if opt_of[i] == j
                and np.min(np.abs(np.asarray(rows) - ys[i])) <= 0.55 * d
            ]

            cx_ref.append(
                float(np.median(selected_x))
                if len(selected_x) >= 3
                else float(col["x"])
            )

        # Construct expected cells.
        cells = []

        for ry in rows:
            row = []

            for cx in cx_ref:
                distances = np.hypot(xs - cx, ys - ry)
                k = int(np.argmin(distances))

                if distances[k] <= 0.70 * d:
                    row.append((
                        float(xs[k]),
                        float(ys[k]),
                        False
                    ))
                else:
                    row.append((float(cx), float(ry), True))

            cells.append(row)

        blocks.append({
            "cols": cx_ref,
            "rows": rows,
            "cells": cells
        })

    return {
        "ok": True,
        "blocks": blocks,
        "d": float(d),
        "pitch": inner
    }


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
    """
    Measure darkness inside OMR bubbles.
    Use local contrast against surrounding paper to improve
    detection of dark marks on pink printed OMR sheets.
    """
    r = layout["d"] / 2.0
    fr = []

    for blk in layout["blocks"]:
        fb = []

        for row in blk["cells"]:
            row_values = []

            for cx, cy, inferred in row:
                value = cell_fraction(gray, cx, cy, r)
                row_values.append(value)

            fb.append(row_values)

        fr.append(np.array(fb, dtype=np.float32))

    if not fr:
        return [], 0.0

    allv = np.concatenate([f.ravel() for f in fr])

    # Estimate the background level from the lighter bubbles.
    base = float(np.percentile(allv, 35))
    base = min(max(base, 0.0), 0.35)

    # Increase contrast between empty and filled bubbles.
    measured = [
        np.clip((f - base) / max(1.0 - base, 0.01), 0.0, 1.0)
        for f in fr
    ]

    return measured, base

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
