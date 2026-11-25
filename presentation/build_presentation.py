"""Build the presentation .pptx from the project's results (python-pptx)."""
import json
import sys
from pathlib import Path

import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
S = json.loads((ROOT / "results/metrics_summary.json").read_text())
TP = S["test_period"]
M = pd.read_csv(ROOT / "results/metrics_by_date.csv")
T = M[M.period == "test"].sort_values("decision_date")
PROF = pd.read_csv(ROOT / "results/shortlist_profiles.csv", index_col=0)
FAC = pd.read_csv(ROOT / "results/shortlist_facility_mix.csv", index_col=0)
SEL = pd.read_csv(ROOT / "results/model_selection.csv", index_col=0)

INK = RGBColor(0x16, 0x21, 0x3A)
BODY = RGBColor(0x3E, 0x4A, 0x5C)
MUTED = RGBColor(0x5F, 0x68, 0x75)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ALT = RGBColor(0xEE, 0xF1, 0xF5)
LINE = RGBColor(0xD3, 0xD9, 0xE2)
ACCENT = RGBColor(0xA8, 0x4F, 0x17)
LOGIT = RGBColor(0x1F, 0x5F, 0xA8)
BASE = RGBColor(0x6B, 0x6F, 0x76)
RANDOM = RGBColor(0xB9, 0xB3, 0xA6)
XGB = RGBColor(0xC0, 0x50, 0x4D)
LIGHT_ON_DARK = RGBColor(0xC9, 0xD3, 0xE3)
HEAD, BODYF = "Cambria", "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W = 13.333
MX = 0.7          # side margin
CW = W - 2 * MX   # content width


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def bg(slide, color):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = color


def text(slide, x, y, w, h, runs, size=16, color=BODY, font=BODYF, bold=False,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, name=None):
    """runs: str or list of paragraphs; a paragraph is str or list of (text, opts)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if i:
            p.space_before = Pt(size * 0.45)
        parts = para if isinstance(para, list) else [(para, {})]
        for t, o in parts:
            r = p.add_run()
            r.text = t
            r.font.size = Pt(o.get("size", size))
            r.font.name = o.get("font", font)
            r.font.bold = o.get("bold", bold)
            r.font.italic = o.get("italic", False)
            r.font.color.rgb = o.get("color", color)
    return tb


def header(slide, eyebrow, title, dark=False):
    text(slide, MX, 0.5, CW, 0.3, eyebrow.upper(), size=13, bold=True,
         color=RGBColor(0xF0, 0xA8, 0x6E) if dark else ACCENT, name="Eyebrow")
    text(slide, MX, 0.85, CW, 1.1, title, size=30, bold=True, font=HEAD,
         color=WHITE if dark else INK, name="Title")


def footer(slide, s, dark=False):
    text(slide, MX, 6.85, CW, 0.3, s, size=11, color=LIGHT_ON_DARK if dark else MUTED, name="Footer")


def card(slide, x, y, w, h, label, body, big=None, fill=WHITE, border=LINE, big_color=INK,
         body_size=14):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = 0.06
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = border
    shp.line.width = Pt(1.25)
    shp.shadow.inherit = False
    yy = y + 0.25
    text(slide, x + 0.25, yy, w - 0.5, 0.3, label.upper(), size=12, bold=True, color=ACCENT)
    yy += 0.38
    if big:
        text(slide, x + 0.25, yy, w - 0.5, 0.7, big, size=36, bold=True, font=HEAD, color=big_color)
        yy += 0.75
    if body:
        text(slide, x + 0.25, yy, w - 0.5, h - (yy - y) - 0.2, body, size=body_size)


def table(slide, x, y, w, rows, col_w, size=13, highlight=None, row_h=0.42):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    tbl = gt.table
    tbl.first_row = True
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(w * cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = INK if i == 0 else (WHITE if i % 2 else ALT)
            c.margin_left = c.margin_right = Inches(0.1)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(size)
            r.font.name = BODYF
            r.font.bold = i == 0 or (highlight is not None and (i, j) in highlight)
            r.font.color.rgb = WHITE if i == 0 else (LOGIT if highlight and (i, j) in highlight else BODY)
    return gt


def notes(slide, s):
    slide.notes_slide.notes_text_frame.text = s


def new(dark=False, alt=False):
    s = prs.slides.add_slide(BLANK)
    bg(s, INK if dark else (ALT if alt else WHITE))
    return s


P100 = TP["precision_at_k"]["100"]
UL = TP["uncertainty_logistic_vs_baseline"]["precision_at_100"]
UV = TP["uncertainty_logistic_vs_baseline"]["violations_in_top_100"]
VIOL = TP["violations_found_in_top_100_per_date"]
lo, hi = UL["ci95_block_bootstrap"]
diff = UL["mean_difference"]

# 1 cover -------------------------------------------------------------------
s = new(dark=True)
text(s, MX, 2.0, CW, 0.35, "MACHINE LEARNING FOR BUSINESS ANALYTICS · TEAM PROJECT", size=14, bold=True,
     color=RGBColor(0xF0, 0xA8, 0x6E))
text(s, MX, 2.5, 10.5, 1.9, "Which restaurants should be inspected first?", size=48, bold=True,
     font=HEAD, color=WHITE)
text(s, MX, 4.45, 10, 0.9, "Choosing 100 routine food inspections in Chicago so that more real failures are found",
     size=20, color=LIGHT_ON_DARK)
text(s, MX, 5.6, CW, 0.4, "[Team member names] · Plaksha MS-AI, Monsoon 2026", size=14, color=LIGHT_ON_DARK)
notes(s, "Open with the problem in one sentence: a city health department can only move 100 places to the "
         "front of its routine inspection queue, and we want those 100 to be the ones most likely to fail. "
         "Introduce the team and who presents which part.")

# 2 decision ----------------------------------------------------------------
s = new()
header(s, "The decision", "A health department can bring only 100 inspections forward each quarter")
cw = (CW - 3 * 0.3) / 4
for i, (lab, body) in enumerate([
        ("Organisation", "A city food-safety department (Chicago’s data, organisation unnamed)"),
        ("Opportunities", "About 6,000 establishments per decision date with a routine visit in the coming year"),
        ("Limit", "K = 100 moved to the front of the routine queue"),
        ("Good decision", "More real food-safety failures found in those 100 visits")]):
    card(s, MX + i * (cw + 0.3), 2.15, cw, 2.3, lab, body, body_size=16)
text(s, MX, 4.85, CW, 0.9, [[("We rank establishments by the chance that their ", {}),
                             ("next routine inspection fails", {"bold": True, "color": INK}),
                             (", and take the top 100.", {})]], size=20)
footer(s, "Outcome observed: the city’s recorded inspection result, not a proxy we invented.")
notes(s, "The brief asks for the organisation, the opportunities, the limit K and what a good decision achieves. "
         "Routine (canvass) inspections are the scheduling decision; complaint and licence inspections are "
         "triggered by events. Guiding question 1: Fail is close to what the department values but not identical; "
         "we add a second metric, violations cited.")

# 3 data ----------------------------------------------------------------------
s = new(alt=True)
header(s, "Data and formulation", "Each row is one establishment at one quarter-end, seen only with what was known then")
cw = (CW - 2 * 0.3) / 3
for i, (big, body) in enumerate([
        (f"{S['dataset']['rows_routine_inspections']:,}", "routine inspections, 2010–2026, City of Chicago Data Portal (dataset 4ijn-s7e5)"),
        ("30", "quarter-end decision dates: 16 for training (2016–19), 14 for evaluation (2022–25)"),
        (f"{S['panel']['train_examples'] + S['panel']['test_examples']:,}", "establishment × date rows, each with 16 history features and a facility type")]):
    card(s, MX + i * (cw + 0.3), 2.15, cw, 2.25, "", body, big=big, body_size=15)
box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(MX), Inches(4.7), Inches(CW), Inches(1.4))
box.adjustments[0] = 0.08
box.fill.solid(); box.fill.fore_color.rgb = INK; box.line.fill.background()
text(s, MX + 0.35, 4.9, CW - 0.7, 1.1, [[("Features", {"bold": True, "color": WHITE}),
     (" use only inspections dated on or before the decision date: past failures, past violation counts, time "
      "since last visit, risk category, facility type, ZIP failure rate. ", {}),
     ("Label", {"bold": True, "color": WHITE}),
     (": did the first routine inspection in the next 12 months fail?", {})]], size=16, color=LIGHT_ON_DARK)
footer(s, "2020–21 is left out as a decision period (COVID disruption) but still feeds the history of later dates.")
notes(s, "Guiding question 3: what was known at the decision date. Every feature is computed from inspections "
         "dated on or before the quarter-end; the label comes strictly after it. Public data, one script "
         "reproduces everything.")

# 4 approaches ----------------------------------------------------------------
s = new()
header(s, "Two approaches on the same decision", "A simple history rule against a transparent model")
cw = (CW - 2 * 0.3) / 3
card(s, MX, 2.15, cw, 3.7, "Approach A · baseline",
     [[("Smoothed past failure rate", {"bold": True, "color": INK, "size": 18})],
      "Inspect the places that failed most often before. The rate is shrunk toward the city average so one bad "
      "visit at a rarely inspected place does not dominate."], body_size=15)
card(s, MX + cw + 0.3, 2.15, cw, 3.7, "Approach B · logistic regression",
     [[("All 16 history features + facility type", {"bold": True, "color": INK, "size": 18})],
      "Adds how bad recent inspections were, time since the last visit, risk category, facility type and ZIP. "
      "Coefficients can be read and explained."], border=LOGIT, body_size=15)
card(s, MX + 2 * (cw + 0.3), 2.15, cw, 3.7, "Challengers",
     [[("15 candidates, picked on 2016–19 only", {"bold": True, "color": INK, "size": 18})],
      "Regularised and spline logistic, a small neural network, Extra Trees and XGBoost. Validation picks the "
      "neural network."], body_size=15)
footer(s, "All approaches score exactly the same candidates at every date, so the comparison is like for like.")
notes(s, "Requirement 4: two defensible approaches. The baseline is the natural simple rule of guiding question 2. "
         "The pipeline also lets 15 candidates compete on 2016–19 validation only; it picks a small neural "
         "network, which ties logistic regression later, so logistic stays approach B because it can be explained.")

# 5 evaluation ----------------------------------------------------------------
s = new(alt=True)
header(s, "How we test it", "Fit and choose on 2016–19, judge on 2022–25")
for i, (lab, body) in enumerate([
        ("1 · Re-order the real queue", "At each date, rank the places the city did visit in the next year. Do the "
         "first 100 contain more failures? This is the design Chicago’s own 2014 pilot used."),
        ("2 · Two metrics", "Precision@100: share of the 100 whose next inspection failed. Violations cited at "
         "those 100 visits: closer to “how much was wrong”."),
        ("3 · No peeking", "Model settings are chosen on pre-2020 dates only, with a 12-month gap so no "
         "inspection labels both a fit row and a validation row.")]):
    card(s, MX + i * (cw + 0.3), 2.15, cw, 2.9, lab, body, body_size=15)
text(s, MX, 5.35, CW, 0.5, "Benchmarks: what 100 random candidates would score, and the history rule.", size=18)
footer(s, "Uncertainty: paired per-date differences with a block bootstrap, because neighbouring quarters share labels.")
notes(s, "Requirement 3: evidence not used to fit or choose the method, against a meaningful baseline. The claim "
         "is about re-ordering inspections the city actually ran. The 2022–25 period was looked at during "
         "development, so we label it exploratory rather than a fresh test.")

# 6 headline (native bar chart) ----------------------------------------------
s = new()
header(s, "Result · 2022–25", "The history rule finds 1.8× random; the model adds about 3 more failures per 100")
cd = CategoryChartData()
names = ["Random 100", "A · history rule", "B · logistic regression", "Neural net (selected)", "XGBoost"]
vals = [TP["mean_base_rate"], P100["baseline"], P100["logistic"], P100["neural"], P100["xgboost"]]
cd.categories = names
cd.add_series("Precision@100", vals)
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(MX), Inches(2.0), Inches(CW), Inches(3.6), cd)
ch = gf.chart
ch.has_legend = False
ch.has_title = False
plot = ch.plots[0]
plot.gap_width = 45
plot.has_data_labels = True
dl = plot.data_labels
dl.number_format = '0.0%'
dl.number_format_is_linked = False
dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size = Pt(16); dl.font.bold = True; dl.font.color.rgb = INK; dl.font.name = BODYF
cols = [RANDOM, BASE, LOGIT, RGBColor(0x7F, 0xA6, 0xD4), XGB]
for i, c in enumerate(cols):
    pt = plot.series[0].points[i]
    pt.format.fill.solid(); pt.format.fill.fore_color.rgb = c
ca = ch.category_axis
ca.reverse_order = True
ca.tick_labels.font.size = Pt(15); ca.tick_labels.font.color.rgb = INK; ca.tick_labels.font.name = BODYF
ca.format.line.color.rgb = LINE
va = ch.value_axis
va.visible = False
va.has_major_gridlines = False
va.maximum_scale = 0.6; va.minimum_scale = 0
text(s, MX, 5.8, CW, 0.8, [[("Logistic vs history rule: ", {}),
     (f"+{diff * 100:.1f} points", {"bold": True, "color": INK}), (", 95% interval ", {}),
     (f"+{lo * 100:.1f} to +{hi * 100:.1f}", {"bold": True, "color": INK}),
     (". Violations cited in the top 100: ", {}),
     (f"{VIOL['logistic']:.0f} vs {VIOL['baseline']:.0f}", {"bold": True, "color": INK}), (" per quarter.", {})]],
     size=18)
footer(s, "Precision@100: share of the 100 shortlisted places whose next routine inspection failed, averaged over 14 decision dates.")
notes(s, f"A simple rule gets most of the way: repeat offenders already give {pct(P100['baseline'])}, almost double "
         f"random. Logistic regression adds {diff * 100:.1f} points, about 3 more failures per 100 inspections, and "
         f"about {UV['mean_difference']:.0f} more cited violations, ahead on all 14 dates. The neural network, which "
         f"validation picked from 15 candidates, scores the same ({pct(P100['neural'])}), so we present the "
         "transparent model. With 14 dates, read the interval as clearly positive, not as a precise size.")

# 7 by date (native column chart) ---------------------------------------------
s = new(alt=True)
d = ((T.prec100_logistic - T.prec100_baseline) * 100).round(0).tolist()
labels = [f"{x[2:4]} Q{(int(x[5:7]) - 1) // 3 + 1}" for x in T.decision_date]
ahead, tied, behind = sum(v > 0 for v in d), sum(v == 0 for v in d), sum(v < 0 for v in d)
header(s, "Is it consistent?", f"Logistic regression is ahead on {ahead} of 14 quarters")
cd = CategoryChartData()
cd.categories = labels
cd.add_series("Logistic minus history rule (points)", d)
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(MX), Inches(1.95), Inches(8.4), Inches(4.6), cd)
ch = gf.chart
ch.has_legend = False
ch.has_title = True
ch.chart_title.text_frame.text = "Logistic minus history rule, Precision@100 (points)"
ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(14)
ch.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = MUTED
ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = False
plot = ch.plots[0]
plot.gap_width = 40
plot.has_data_labels = True
plot.data_labels.font.size = Pt(12); plot.data_labels.font.color.rgb = INK
plot.data_labels.number_format = '+0;−0;0'; plot.data_labels.number_format_is_linked = False
plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
plot.series[0].invert_if_negative = False
for i, v in enumerate(d):
    pt = plot.series[0].points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = LOGIT if v > 0 else (BASE if v == 0 else ACCENT)
from pptx.oxml.ns import qn as _qn
for dpt in plot.series[0]._element.findall(_qn("c:dPt")):
    if dpt.find(_qn("c:invertIfNegative")) is None:
        inv = dpt.makeelement(_qn("c:invertIfNegative"), {"val": "0"})
        dpt.insert(1, inv)  # after c:idx, as the schema requires
ca = ch.category_axis
ca.tick_labels.font.size = Pt(11); ca.tick_labels.font.color.rgb = BODY
from pptx.enum.chart import XL_TICK_LABEL_POSITION
ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW
va = ch.value_axis
va.has_major_gridlines = True
va.major_gridlines.format.line.color.rgb = LINE
va.tick_labels.font.size = Pt(11); va.tick_labels.font.color.rgb = MUTED
va.format.line.fill.background()
x2 = MX + 8.7
card(s, x2, 1.95, CW - 8.7, 1.3, "Ahead / tied / behind", None, big=f"{ahead} / {tied} / {behind}")
card(s, x2, 3.4, CW - 8.7, 1.3, "Violations cited", None,
     big=f"{UV['dates_ahead']} / {UV['dates_tied']} / {UV['dates_behind']}")
card(s, x2, 4.85, CW - 8.7, 1.7, "Honest caveat",
     "Neighbouring quarters share labels, so these 14 are not independent. The lead narrows in the latest quarters.",
     body_size=14)
notes(s, f"A sign test on {ahead} wins and {behind} losses gives p about "
         f"{UL['sign_test_p_if_dates_independent']:.2f}, but it assumes independent dates, which they are not: a "
         "place’s next inspection can be the label for up to four consecutive quarters. That is why we report the "
         "block-bootstrap interval. The lead narrows in late 2024 and 2025: possible drift, a reason for rolling "
         "revalidation.")

# 8 precision at K ------------------------------------------------------------
s = new()
header(s, "Does it hold for other K?", "For very short lists the history rule wins; from 100 up the models do")
ks = ["25", "50", "100", "150", "200", "300"]
pk = TP["precision_at_k"]
rows = [["Precision@K, 2022–25 mean"] + [f"K = {k}" for k in ks]]
order = [("A · history rule", "baseline"), ("B · logistic regression", "logistic"),
         ("Neural net (selected)", "neural"), ("XGBoost", "xgboost")]
hl = set()
for i, (lab, a) in enumerate(order, start=1):
    rows.append([lab] + [pct(pk[k][a]) for k in ks])
for j, k in enumerate(ks, start=1):
    best = max(order, key=lambda t: pk[k][t[1]])
    hl.add((order.index(best) + 1, j))
table(s, MX, 2.1, CW, rows, [0.28] + [0.12] * 6, size=15, highlight=hl, row_h=0.55)
box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(MX), Inches(5.15), Inches(CW), Inches(1.2))
box.adjustments[0] = 0.08
box.fill.solid(); box.fill.fore_color.rgb = INK; box.line.fill.background()
text(s, MX + 0.35, 5.3, CW - 0.7, 0.9, "The top 25–50 by failure history are the surest bets, and the rule already "
     "finds them. The models earn their keep further down the list, where they spot places the rule misses.",
     size=17, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
footer(s, f"Bold blue = best at that K. Random selection would find {pct(TP['mean_base_rate'])} at every K.")
notes(s, "Guiding question 4, for which problems is the approach fine? If the department could only bring forward "
         "25 or 50 places, the simple rule is as good or better. At 100 and beyond the models are ahead by 3 to 5 "
         "points.")

# 9 overlap -------------------------------------------------------------------
s = new(dark=True)
ov = TP["mean_overlap"]
header(s, "How the shortlists differ", f"The two top-100 lists share only about {ov['baseline_logistic']:.0f} places", dark=True)
text(s, MX, 2.2, 6.2, 2.0, f"{ov['baseline_logistic']:.1f}", size=110, bold=True, font=HEAD, color=WHITE)
text(s, MX, 4.4, 6.0, 1.2, "of 100 places on average appear on both the history-rule and the logistic shortlists, "
     "across the 14 quarters.", size=18, color=LIGHT_ON_DARK)
items = [(ov["baseline_logistic"], "history rule and logistic"), (ov["baseline_xgboost"], "history rule and XGBoost"),
         (ov["logistic_xgboost"], "logistic and XGBoost"), (TP["chance_overlap"], "expected by pure chance")]
text(s, 7.6, 2.25, 5.0, 0.4, "Average shared places per quarter", size=14, color=LIGHT_ON_DARK)
for i, (v, lab) in enumerate(items):
    text(s, 7.6, 2.8 + i * 0.8, 5.0, 0.6, [[(f"{v:.1f}", {"bold": True, "color": WHITE, "size": 24}),
                                            (f"   {lab}", {})]], size=18, color=LIGHT_ON_DARK)
footer(s, "About 72 of each list’s 100 places are picks the other approach would not make.", dark=True)
notes(s, "Requirement 4: show how the shortlists differ. Far more overlap than chance, so both find the same "
         "risky core, but most picks differ. The two models agree with each other more than either agrees with "
         "the rule, so the extra features drive the difference.")

# 10 why ----------------------------------------------------------------------
s = new()
header(s, "Why they differ", "The rule picks chronic failers; the model picks places whose recent visits were bad")
g = ["baseline only", "both", "logistic only", "all candidates"]
def prow(label, key, fmt):
    return [label] + [fmt(PROF.loc[key, c]) for c in g]
rows = [["Mean per establishment, 2022–25", "History rule only", "Both", "Logistic only", "All candidates"],
        prow("Prior routine inspections", "prior routine inspections", lambda v: f"{v:.1f}"),
        prow("Prior failures", "prior failures", lambda v: f"{v:.1f}"),
        prow("Violations, last 3 inspections", "mean violations, last 3 inspections", lambda v: f"{v:.1f}"),
        ["Schools / grocery stores"] + [f"{FAC.loc['School', c] * 100:.0f}% / {FAC.loc['Grocery Store', c] * 100:.0f}%" for c in g],
        prow("Next inspection failed", "share whose next inspection FAILED", lambda v: f"{v * 100:.0f}%")]
table(s, MX, 2.05, CW, rows, [0.36, 0.16, 0.16, 0.16, 0.16], size=15, highlight={(3, 3), (5, 3)}, row_h=0.5)
text(s, MX, 5.25, CW, 1.2, [[("The rule only sees ", {}), ("how often", {"bold": True, "color": INK}),
     (" a place failed, so it keeps long-history schools whose recent visits are fairly clean. The model also "
      "weighs ", {}), ("how bad the last visits were", {"bold": True, "color": INK}),
     (", and swaps them for restaurants and grocery stores that are getting worse.", {})]], size=17)
notes(s, "The explain-why part of requirement 4. The model’s extra picks have fewer failures on record but about "
         "15 violations at recent visits, and they fail 43% of the time against 38% for the rule’s extra picks. "
         "Places strong on both counts appear in both lists and fail 59% of the time. Source: "
         "results/shortlist_profiles.csv.")

# 11 model choice -------------------------------------------------------------
s = new(alt=True)
header(s, "Choosing the model honestly", "Without a 12-month gap, validation rewards the model that memorises")
crit = "criterion (mean of purged)"
leaky = [c for c in SEL.columns if c.startswith("U1")][0]
def lk(name):
    v = SEL.loc[name, leaky]
    return "not run" if pd.isna(v) else pct(v)
rows = [["Precision@100", "Leaky check (no gap)", "Purged check (12-month gap)", "2022–25"],
        ["Neural net 16-8 (selected)", lk("neural 16-8 alpha=10"), pct(SEL.loc["neural 16-8 alpha=10", crit]) + " (best)", pct(P100["neural"])],
        ["Logistic regression (approach B)", lk("logistic (reference)"), pct(SEL.loc["logistic (reference)", crit]), pct(P100["logistic"])],
        ["XGBoost, depth 6, 500 trees", lk("d6-500-mcw5") + " (looks best)", pct(SEL.loc["d6-500-mcw5", crit]), "not used"],
        ["History rule (approach A)", lk("baseline"), pct(SEL.loc["baseline", crit]), pct(P100["baseline"])]]
table(s, MX, 2.05, CW, rows, [0.37, 0.21, 0.24, 0.18], size=15, highlight={(1, 2), (3, 1)}, row_h=0.5)
box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(MX), Inches(4.85), Inches(CW), Inches(1.5))
box.adjustments[0] = 0.08
box.fill.solid(); box.fill.fore_color.rgb = INK; box.line.fill.background()
ins = S["train_period_in_sample"]["precision_at_100"]
text(s, MX + 0.35, 5.0, CW - 0.7, 1.2, "One inspection is the label for about 3 consecutive quarterly rows. Without a "
     "gap, validation sees labels it trained on. With the gap, 15 candidates compete and the smooth models win; "
     f"XGBoost scores {pct(ins['xgboost'], 0)} in-sample but only {pct(P100['xgboost'], 0)} later.",
     size=17, color=WHITE, anchor=MSO_ANCHOR.MIDDLE)
footer(s, "Leaky: fit 2016–18 → validate 2019. Purged: mean of fit 2016–17 → 2019 and fit 2016 → 2018.")
notes(s, "Quarterly rows reuse the same next inspection, so a validation split without a gap equal to the 12-month "
         "label window rewards memorisation. With the gap, a small neural network and the logistic models lead. "
         "The neural network was selected by about 2 points on only eight validation dates and ties logistic "
         "regression on 2022–25, so we present the transparent model.")

# 12 limits -------------------------------------------------------------------
s = new()
header(s, "What this establishes, and what it cannot", "A better order for the inspections the city runs; not a city-wide risk map")
lw = 5.4
card(s, MX, 2.15, lw, 3.9, "Established",
     "Using only what was known at each date, the model re-orders the real routine queue so that about 3 more of "
     "every 100 early visits find a failure, and more violations, over three post-pandemic years.", body_size=16)
card(s, MX + lw + 0.3, 2.15, CW - lw - 0.3, 3.9, "Cannot establish",
     [[("Uninspected places: ", {"bold": True, "color": INK}), ("their outcome never exists (selective labels).", {})],
      [("Fail ≠ unsafe: ", {"bold": True, "color": INK}), ("inspector strictness shapes results; the data has no inspector ID.", {})],
      [("Regime change: ", {"bold": True, "color": INK}), ("the 2018 food code (45 → 63 violations) and COVID shifted the data.", {})],
      [("Small sample: ", {"bold": True, "color": INK}), ("14 quarters that share labels.", {})]], body_size=16)
footer(s, "Related work: Chicago CDPH 2014 pilot; Kannan et al. 2019; Glaeser et al. 2016; Lakkaraju et al. 2017.")
notes(s, "Guiding questions 3 and 4. We only see outcomes where the city inspected, so the claim is about "
         "re-ordering. Kannan et al. found Chicago’s own model leaned on which inspectors last visited. Ten of the "
         "16 training dates predate the July 2018 food code. Where it may look fine but not help: very short "
         "lists, places with no history, and places the city never visits.")

# 13 next ---------------------------------------------------------------------
s = new(alt=True)
header(s, "What we tried next, and what it suggests", "Complaint history helps; rank by harm averted per inspector-hour next")
nr = json.loads((ROOT / "results/experiment_nonroutine.json").read_text())
ev = nr["exploratory_test"]
cw = (CW - 3 * 0.3) / 4
card(s, MX, 2.15, cw, 2.4, "Tested: complaints",
     f"Complaint and re-inspection history: logistic {pct(ev['logistic_original'])} → {pct(ev['logistic_nonroutine'])}. "
     "To confirm on a fresh period.", body_size=14, border=LOGIT)
card(s, MX + (cw + 0.3), 2.15, cw, 2.4, "Tested: no gain",
     "A model trained directly for ranking, violation-topic features and the Kumo Small tabular model.", body_size=14)
card(s, MX + 2 * (cw + 0.3), 2.15, cw, 2.4, "Next: severity × exposure",
     "Weight Priority violations, and schools, day care and care homes, more heavily.", body_size=14)
card(s, MX + 3 * (cw + 0.3), 2.15, cw, 2.4, "Next: cost and exploration",
     "Divide by visit and travel time; keep 10–15% of slots random so the model keeps learning.", body_size=14)
text(s, MX, 4.9, CW, 1.2, "With equal weights, no travel and no exploration, the proposal reduces to today’s "
     "Precision@100 ranking, so it extends the project rather than replacing it.", size=18)
footer(s, "Experiments: results/experiment_*.json. Exploration idea from audit-selection work (Henderson et al. 2023).")
notes(s, "The complaint-history gain was found after the 2022–25 period had been looked at, so present it as a "
         "lead, not a result. The business extension: inspector time is the scarce resource; a failure at a school "
         "kitchen matters more than one at a low-traffic kiosk; and a purely greedy ranking stops generating data "
         "about places it never picks.")

# 14 close --------------------------------------------------------------------
s = new(dark=True)
text(s, MX, 1.3, CW, 0.35, "SUMMARY", size=14, bold=True, color=RGBColor(0xF0, 0xA8, 0x6E))
text(s, MX, 1.75, CW, 1.2, "Past failures get you most of the way; recent violations add the rest", size=32,
     bold=True, font=HEAD, color=WHITE)
pts = [f"The history rule finds {pct(P100['baseline'], 0)} failures in the top 100, against "
       f"{pct(TP['mean_base_rate'], 0)} at random.",
       f"Logistic regression adds {diff * 100:.1f} points (95% interval +{lo * 100:.1f} to +{hi * 100:.1f}) and "
       "more violations on every quarter; the validation-selected neural net does the same.",
       "The lists differ because the model looks at how bad recent visits were, not just how often a place failed.",
       "Most promising next feature: complaint and re-inspection history, to be confirmed."]
tb = text(s, MX, 3.15, CW, 2.8, pts, size=19, color=RGBColor(0xE9, 0xED, 0xF4))
for p in tb.text_frame.paragraphs:
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(Inches(0.3))); pPr.set("indent", str(-Inches(0.3)))
    from pptx.oxml.ns import qn
    for tag in ("a:buNone", "a:buChar"):
        for el in pPr.findall(qn(tag)):
            pPr.remove(el)
    bu = pPr.makeelement(qn("a:buChar"), {"char": "•"})
    pPr.append(bu)
text(s, MX, 6.3, CW, 0.4, "Reproduce: python run_project.py · data: City of Chicago Food Inspections (4ijn-s7e5) · "
     "github.com/ajayc-iima/Food-Inspections", size=12, color=LIGHT_ON_DARK)
notes(s, "Close by answering the brief’s central question: problem, data, method, evaluation and conclusions fit "
         "together. Likely questions: why the interval and not a t-test (dates share labels); why present logistic "
         "when the neural net was selected (they tie on 2022–25 and logistic is explainable); did a ranking model or "
         "Kumo help (no).")

prs.core_properties.title = "Which restaurants should be inspected first?"
OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(OUT)
print("saved", OUT, len(prs.slides), "slides")
