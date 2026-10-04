"""viz.py - image rendering helpers (PIL only, no Streamlit)."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

AMBER = (255, 190, 0)
RED = (255, 70, 70)

# outcome -> (colour, tag shown on the feed)
OUTCOME_STYLE = {
    "NEUTRALISED":       ((0, 230, 200), "NEUTRALISED"),
    "CLEARED":           ((90, 210, 130), "BIRD"),
    "FALSE ALARM":       ((90, 210, 130), "BIRD"),
    "STOOD DOWN":        ((90, 160, 255), "FRIENDLY"),
    "ARRIVED SAFELY":    ((90, 160, 255), "FRIENDLY"),
    "FALSE ALERT":       ((90, 160, 255), "FRIENDLY"),
    "MONITORED":         ((255, 120, 120), "MONITORED"),
    "IMPACT":            ((255, 120, 0), "IMPACT"),
    "IMPACT MITIGATED":  ((255, 150, 0), "IMPACT"),
    "INTEL COMPROMISED": ((255, 120, 0), "COMPROMISED"),
    "BLUE-ON-BLUE":      ((210, 90, 255), "FRIENDLY LOST"),
}


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def load_display(path: str, width: int = 1600) -> Image.Image:
    """Load a (possibly 4K) frame downscaled for the feed."""
    with Image.open(path) as im:
        try:
            im.draft("RGB", (width, int(width * im.height / im.width)))
        except Exception:
            pass
        im = im.convert("RGB")
        if im.width > width:
            im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
        return im


def crop_zoom(path: str, xywhn, zoom: float, out: int = 560) -> Image.Image:
    """Crop the ORIGINAL full-resolution frame around a contact (optic zoom)."""
    x, y, w, h = xywhn
    with Image.open(path) as im:
        im = im.convert("RGB")
        W, H = im.size
        side = int(max(max(w * W, h * H) * zoom, 160))
        side = min(side, W, H)
        cx, cy = x * W, y * H
        x0 = int(min(max(cx - side / 2, 0), W - side))
        y0 = int(min(max(cy - side / 2, 0), H - side))
        crop = im.crop((x0, y0, x0 + side, y0 + side)).resize((out, out), Image.LANCZOS)
    d = ImageDraw.Draw(crop, "RGBA")
    m, L = out // 2, 14
    for sx, sy in ((m - 40, m - 40), (m + 40, m - 40), (m - 40, m + 40), (m + 40, m + 40)):
        dx = L if sx < m else -L
        dy = L if sy < m else -L
        d.line([(sx, sy), (sx + dx, sy)], fill=AMBER + (220,), width=2)
        d.line([(sx, sy), (sx, sy + dy)], fill=AMBER + (220,), width=2)
    return crop


def _style(c):
    if c["status"] == "OPEN":
        return AMBER, "UNKNOWN"
    if c["status"] == "CLASSIFIED":
        return RED, "DRONE"
    return OUTCOME_STYLE.get(c["outcome"], ((200, 200, 200), str(c["outcome"])))


def render_feed(base: Image.Image, frame, M, sel: int) -> Image.Image:
    """Annotated sensor feed: AI cues every contact; colour shows your decision state."""
    img = base.copy()
    W, H = img.size
    d = ImageDraw.Draw(img, "RGBA")
    f = _font(max(15, W // 70))

    for i, c in enumerate(frame["contacts"]):
        x, y, w, h = c["xywhn"]
        bw, bh = max(w * W, 56), max(h * H, 56)
        x0, y0, x1, y1 = x * W - bw / 2, y * H - bh / 2, x * W + bw / 2, y * H + bh / 2
        col, tag = _style(c)
        if i == sel:
            d.rectangle([x0 - 7, y0 - 7, x1 + 7, y1 + 7], outline=(255, 255, 255, 255), width=2)
        d.rectangle([x0, y0, x1, y1], outline=col + (255,), width=3)
        label = f"{c['cid']} {tag}"
        tb = d.textbbox((0, 0), label, font=f)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        d.rectangle([x0, y0 - th - 10, x0 + tw + 10, y0], fill=(0, 0, 0, 185))
        d.text((x0 + 5, y0 - th - 7), label, fill=col + (255,), font=f)

    # header strip
    bar = int(H * 0.055)
    d.rectangle([0, 0, W, bar], fill=(0, 0, 0, 170))
    txt = (f"EO CAM-01   |   FRAME {M['fi'] + 1}/{len(M['frames'])}   |   {M['roe'].upper()}"
           f"   |   ASSET: {frame['asset'].upper()}" + ("   |   CIVILIANS NEAR" if frame["civilians"] else ""))
    d.text((14, bar * 0.2), txt, fill=(120, 255, 160, 255), font=f)
    return img


def render_gt(base: Image.Image, boxes, W: int, H: int) -> Image.Image:
    """Ground-truth view for the dataset tab: red = drone, green = bird."""
    img = base.copy()
    bw_, bh_ = img.size
    d = ImageDraw.Draw(img, "RGBA")
    f = _font(max(14, bw_ // 60))
    for cls, x, y, w, h in boxes:
        col = RED if cls == 0 else (90, 220, 130)
        bw, bh = max(w * bw_, 40), max(h * bh_, 40)
        x0, y0 = x * bw_ - bw / 2, y * bh_ - bh / 2
        d.rectangle([x0, y0, x0 + bw, y0 + bh], outline=col + (255,), width=3)
        d.text((x0, y0 - f.size - 4 if hasattr(f, "size") else y0 - 18),
               "drone" if cls == 0 else "bird", fill=col + (255,), font=f)
    return img
