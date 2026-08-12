"""
Generate the boot self-test page. Run at BUILD time:  python make_selftest_page.py OUT.png

WHY THIS EXISTS
    The self-test used to bake a real client audit page. That forced the image to stay
    private, which forced a GitHub PAT and a RunPod registry credential — two setup
    steps whose only job was protecting one file. Generating the page removes the client
    data, the image can be public, and both steps disappear.

WHY IT IS NOT THE OLD TOY IMAGE
    The Replicate build self-tested with a 400x120 synthetic image. It passed on every
    boot while real 3166x4096 pages came back empty, because at that size nothing
    stresses VRAM and the layout model has almost nothing to do. This page is FULL SIZE
    and full of blocks, so it walks the same code path and allocates comparable memory.

WHAT IT PROVES (and what it doesn't)
    Proves the worker booted correctly: models present, layout labels intact, bbox
    scaling applied, no CUDA OOM. Those are exactly the failure modes that silently
    return empty markdown.
    Does not prove OCR accuracy on real scans — a boot check never could. That is what
    the ladder and the golden-sample comparison are for.
"""
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 3166, 4096          # same as the real SplitImage pages (~375 dpi A4)
MARGIN = 260
BODY_PT, HEAD_PT = 52, 68  # ~10pt / ~13pt at this resolution
LINE_GAP, HEAD_GAP = 78, 118

# The handler asserts these words come back from OCR. A minimum-length threshold would
# need re-tuning every time this page changes; a sentinel is self-calibrating.
SENTINEL_WORDS = ("SELFTEST", "SENTINEL", "BRAVO")

FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/usr/share/fonts/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
    (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),   # local preview
]

PARAS = [
    "The accompanying notes form an integral part of these financial statements and "
    "should be read together with the report of the auditors.",
    "All amounts are stated in Ringgit Malaysia unless otherwise indicated in the "
    "relevant note to the accounts set out on the following pages.",
    "These financial statements have been prepared in accordance with the applicable "
    "approved accounting standards and the provisions of the Companies Act.",
    "The preparation of financial statements requires the use of estimates and "
    "assumptions that affect the reported amounts of assets and liabilities.",
    "Revenue is measured at the fair value of the consideration received or receivable, "
    "net of returns, trade discounts and volume rebates.",
    "Property, plant and equipment are stated at cost less accumulated depreciation and "
    "any accumulated impairment losses recognised to date.",
    "Before the statements were made out, the Directors took reasonable steps to "
    "ascertain that action had been taken in relation to doubtful debts.",
    "There were no material transfers to or from reserves and provisions during the "
    "financial year under review except as disclosed in the accounts.",
    "No dividend has been paid or declared by the Company since the end of the previous "
    "financial year ended on the reporting date stated above.",
]

HEADINGS = [
    "Statement Of Comprehensive Income",
    "Basis Of Preparation",
    "Significant Accounting Policies",
    "Other Statutory Information",
    "Reserves And Provisions",
]

TABLE = [
    ["Financial Results", "2025", "2024"],
    ["Revenue", "4,182,905", "3,776,140"],
    ["Profit before taxation", "612,374", "540,118"],
    ["Taxation", "(147,929)", "(129,628)"],
    ["Profit for the year", "464,445", "410,490"],
]


def load_fonts():
    """A silent fallback to PIL's bitmap font renders unreadably small text at this page
    size — a self-test page that cannot be OCR'd would fail every boot for the wrong
    reason. Refuse to generate instead."""
    for regular, bold in FONT_CANDIDATES:
        try:
            return (ImageFont.truetype(regular, BODY_PT),
                    ImageFont.truetype(bold, HEAD_PT))
        except Exception:
            continue
    raise SystemExit(
        "no TrueType font found (tried: %s). In the image, install fonts-dejavu-core."
        % ", ".join(p[0] for p in FONT_CANDIDATES))


def draw_table(d, y, f_body, f_head):
    """A real bordered table, so the layout detector must emit native_label='table'."""
    width = W - 2 * MARGIN
    col_x = [MARGIN + 30, MARGIN + int(width * 0.58), MARGIN + int(width * 0.80)]
    row_h = 104
    for r, row in enumerate(TABLE):
        ry = y + r * row_h
        d.rectangle([MARGIN, ry, MARGIN + width, ry + row_h], outline="black", width=4)
        for c, cell in enumerate(row):
            d.text((col_x[c], ry + 22), cell,
                   font=(f_head if r == 0 else f_body), fill="black")
    return y + len(TABLE) * row_h


def wrap(d, text, font, max_w):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if d.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def main(out_path):
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f_body, f_head = load_fonts()
    max_w = W - 2 * MARGIN
    bottom = H - MARGIN

    y = MARGIN
    table_done = False
    hi = pi = 0
    blocks = 0

    # Fill the page: heading, two paragraphs, repeat; table once in the middle.
    while y < bottom - 200:
        d.text((MARGIN, y), HEADINGS[hi % len(HEADINGS)], font=f_head, fill="black")
        y += HEAD_GAP
        hi += 1
        blocks += 1

        for _ in range(2):
            for line in wrap(d, PARAS[pi % len(PARAS)], f_body, max_w):
                if y > bottom - 140:
                    break
                d.text((MARGIN, y), line, font=f_body, fill="black")
                y += LINE_GAP
            y += 34
            pi += 1
            blocks += 1

        if not table_done and y > H * 0.35:
            if y + len(TABLE) * 104 < bottom - 200:
                y = draw_table(d, y + 30, f_body, f_head) + 60
                table_done = True
                blocks += 1

    # The sentinel goes last so it also proves the tail of the page was transcribed.
    d.text((MARGIN, min(y, bottom - 80)),
           " ".join(SENTINEL_WORDS) + " 20260812 CHARLIE DELTA ECHO",
           font=f_head, fill="black")

    img.save(out_path, format="PNG")
    print("selftest page: %s  %dx%d  blocks~%d  table=%s  sentinel=%s"
          % (out_path, W, H, blocks + 1, table_done, " ".join(SENTINEL_WORDS)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "selftest_page.png")
