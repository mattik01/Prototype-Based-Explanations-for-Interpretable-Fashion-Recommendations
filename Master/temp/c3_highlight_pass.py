"""C3 (Rudin 2019) reading-prioritization highlight pass.
Green = core/read this; Yellow = secondary/if time. Whole-passage word-box ranges.
Builds on a COPY, never the original."""
import fitz, shutil, os
from collections import OrderedDict

SRC = "Master/literature/papers/C_prototype_interpretability/C3_rudin_2019_stop_explaining_black_boxes.pdf"
DST = "Master/literature/papers/C_prototype_interpretability/C3_rudin_2019_stop_explaining_black_boxes-annotated.pdf"

GREEN = (0.62, 0.92, 0.55)
YELLOW = (1.0, 0.90, 0.42)
OP = 0.40

# (page, start_phrase, end_phrase, color, label)
P = [
    # --- GREEN core ---
    (0, "Rather than trying to create models", "model actually computes", GREEN, "G1 intrinsic>posthoc thesis"),
    (0, "Interpretability is a domain", "rather than individually", GREEN, "G2 interpretability defn / sparsity / case-based"),
    (1, "It is a myth that there is necessarily", "naturally meaningful features", GREEN, "G3 trade-off myth"),
    (2, "Explainable ML methods provide explanations that are not faithful", "only the explanation", GREEN, "G4 explanations must be wrong"),
    (3, "Saliency maps can be useful", "with that part of the image", GREEN, "G5 saliency where vs what"),
    (11, "constitutes interpretability by considering", "real, and not posthoc", GREEN, "G6 part-based reasoning defn"),
    (11, "Chen, Li, and colleagues have been building", "prototypical feathers of a blue jay", GREEN, "G7 ProtoPNet mechanism"),
    (12, "metric between parts of images", "are not posthoc explanations", GREEN, "G8 explanations are actual computations"),
    (13, "Here is the Rashomon set argument", "interpretable and accurate", GREEN, "G9 Rashomon set argument"),
    # --- YELLOW secondary ---
    (2, "An explainable model that has a 90%", "either the explanation or the original", YELLOW, "Y1 90/10 trust"),
    (3, "Let us stop calling approximations", "black box model predictions explanations", YELLOW, "Y2 stop calling them explanations"),
    (5, "Corporations can make", "were used instead", YELLOW, "Y3a profit incentive"),
    (5, "The COMPAS model is equally accurate", "shown in Figure 3", YELLOW, "Y3b COMPAS=CORELS parity"),
    (12, "Training this prototype network is not as easy", "before the prototype layer was added", YELLOW, "Y4 prototype-net accuracy parity"),
    (13, "Dimension reduction to interpretable dimensions", "is an important theme", YELLOW, "Y5 dimension reduction theme"),
    (14, "If this commentary can shift the focus", "considered this document a success", YELLOW, "Y6 conclusion thesis"),
    (19, "It is possible to create a global model", "very simple to calculate", YELLOW, "Y7 smaller-than-global explanation"),
]


def highlight(page, start, end, color, label):
    s_hits = page.search_for(start)
    e_hits = page.search_for(end)
    if not s_hits:
        print(f"  !! START not found: {label} | {start!r}")
        return 0
    if not e_hits:
        print(f"  !! END not found:   {label} | {end!r}")
        return 0
    s = s_hits[0]
    after = [r for r in e_hits if r.y0 >= s.y0 - 1]
    e = after[0] if after else e_hits[-1]
    sel = []
    for w in page.get_text("words"):
        wx0, wy0, wx1, wy1 = w[0], w[1], w[2], w[3]
        if wy0 > s.y0 + 2:
            aft = True
        elif abs(wy0 - s.y0) <= 2:
            aft = wx0 >= s.x0 - 0.5
        else:
            aft = False
        if wy1 < e.y1 - 2:
            bef = True
        elif abs(wy1 - e.y1) <= 2:
            bef = wx1 <= e.x1 + 0.5
        else:
            bef = False
        if aft and bef:
            sel.append(w)
    if not sel:
        print(f"  !! no words selected: {label}")
        return 0
    lines = OrderedDict()
    for w in sel:
        lines.setdefault((w[5], w[6]), []).append(w)
    n = 0
    for ws in lines.values():
        r = fitz.Rect(min(w[0] for w in ws), min(w[1] for w in ws),
                      max(w[2] for w in ws), max(w[3] for w in ws))
        a = page.add_highlight_annot(r)
        a.set_colors(stroke=color)
        a.set_opacity(OP)
        a.update()
        n += 1
    return n


shutil.copy(SRC, DST)
doc = fitz.open(DST)
total = 0
for pno, start, end, color, label in P:
    c = "GRN" if color is GREEN else "YEL"
    n = highlight(doc[pno], start, end, color, label)
    print(f"[{c}] p{pno:>2} lines={n:>2}  {label}")
    total += n
doc.save(DST + ".out", garbage=4, deflate=True)
doc.close()
os.replace(DST + ".out", DST)
print(f"\nTotal highlight rects: {total}")
print(f"Saved: {DST} ({os.path.getsize(DST)//1024} KB)")
