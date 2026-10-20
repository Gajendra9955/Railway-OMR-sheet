"""File loading: PDF / JPG / PNG -> list of BGR numpy pages."""
import io

import cv2
import numpy as np
from PIL import Image, ImageOps

MAX_PAGES = 30
Image.MAX_IMAGE_PIXELS = 80_000_000


class FileProblem(Exception):
    """A problem with an uploaded file that we can explain to the user."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def sniff(data: bytes):
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return None


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def load_pages(data: bytes, dpi: int = 200, max_pages: int = MAX_PAGES):
    """Return (kind, [BGR page images]). Raises FileProblem on bad files."""
    kind = sniff(data)
    if kind is None:
        raise FileProblem(
            "bad_format",
            "यह फ़ाइल PDF, JPG या PNG नहीं लगती। कृपया सही फ़ाइल चुनें।",
        )
    if kind == "pdf":
        from pdf2image import convert_from_bytes, pdfinfo_from_bytes

        try:
            info = pdfinfo_from_bytes(data)
            n = int(info.get("Pages", 1))
        except Exception:
            raise FileProblem("bad_pdf", "PDF खुल नहीं पा रही है (फ़ाइल खराब या पासवर्ड वाली हो सकती है)।")
        if n > max_pages:
            raise FileProblem("too_many_pages", f"PDF में {n} पेज हैं; अधिकतम {max_pages} पेज समर्थित हैं।")
        try:
            pages = convert_from_bytes(data, dpi=dpi, first_page=1, last_page=n)
        except Exception:
            raise FileProblem("bad_pdf", "PDF को चित्र में बदला नहीं जा सका।")
        return kind, [pil_to_bgr(p) for p in pages]
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        raise FileProblem("bad_image", "चित्र खुल नहीं पा रहा है।")
    return kind, [pil_to_bgr(img)]
