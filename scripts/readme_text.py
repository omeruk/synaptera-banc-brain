"""The repository README, the figure gallery and the list of every number they print.

    env -u PYTHONPATH python scripts/readme_text.py [--overwrite]
    env -u PYTHONPATH python scripts/readme_text.py --check      # exit 1 if a committed output differs; writes nothing

Writes README.md, docs/FIGURES.md and docs/readme_numbers.json. No number is typed here (CLAUDE.md rule 11): each one
is read from a committed JSON (docs/results/*.json, docs/figures/*.json, configs/f5_splits.json) or a committed CSV
(docs/figures/input_checks.csv, loaded as {"rows": [...]} by csv.DictReader, so its key is "rows"), from a code constant
(parsed from the source, not imported), or from the text of a pre-registration, and docs/readme_numbers.json records
for each printed string the file(s) it appears in and its source. References and the commands of the Reproduce
section are copied from repository files and checked against them. Refuses to overwrite an existing output without
--overwrite. scripts/verify_report.py (check_readme) recomputes every entry of docs/readme_numbers.json.
"""
import argparse
import ast
import csv
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README, GALLERY, NUMBERS = ROOT / "README.md", ROOT / "docs" / "FIGURES.md", ROOT / "docs" / "readme_numbers.json"
REPORT = "docs/results/f5_stageA_report.json"
POSTHOC = "docs/results/f5_stageA_posthoc.json"
VALID = "docs/results/f5_stageA_validation.json"
CURVES = "docs/results/f5_stageA_curves.json"
RUNNER = "docs/results/f5_stageA_runner.json"
RECORDS = "docs/results/f5_stageA_records.json"
SUMMARY = "docs/results/summary_f0_f3.json"
XWALK = "docs/results/f3_dn_crosswalk.json"
F3B3 = "docs/results/f3b3_specificity.json"
RATE_LIF = "docs/results/f5_rate_lif.json"
SIDE = "docs/results/side_audit.json"
SIDE_POST = "docs/results/side_audit_post_audit.json"
BENCH = "docs/results/f5_benchmark.json"
SPLITS = "configs/f5_splits.json"
TRANSFORMS = "docs/figures/transforms.json"
WARP = "docs/figures/warp_check.json"
VI, ICSV = "docs/figures/visual_input.json", "docs/figures/input_checks.csv"
P19 = "prereg/19_f5_gpu_reproducibility_bridge_units_readout.md"
P14, P21 = "prereg/14_f5_training.md", "prereg/21_f5_stageA_posthoc_plan.md"
REPO_URL = "https://github.com/omeruk/synaptera-banc-brain"
AUTHOR = "Ömer Faruk Topal, Department of Geomatics Engineering, Istanbul Technical University."
CREDIT = "Designed and directed by Ömer Faruk Topal; implemented with the help of Claude Code (Anthropic)."
LR_MIN_DECIMALS = 4            # learning-rate grid table: at least this many decimals, more where an arm's values would tie

ENTRIES = []
CUR = {"file": None}
_JSON = {}


def _load(f):
    """A committed JSON, or a committed CSV as {"rows": [one dict per row]}."""
    if f not in _JSON:
        if f.endswith(".csv"):
            with open(ROOT / f, newline="") as fh:
                _JSON[f] = {"rows": list(csv.DictReader(fh))}
        else:
            _JSON[f] = json.loads((ROOT / f).read_text())
    return _JSON[f]


def _get(f, path):
    v = _load(f)
    for k in path.split("/"):
        v = v[int(k)] if isinstance(v, list) else v[k]
    return v


def _rec(s, source, kind):
    ENTRIES.append(dict(string=s, file=CUR["file"], kind=kind, source=source))
    return s


def J(f, path, fmt="{}"):
    """A value of a committed JSON."""
    return _rec(fmt.format(_get(f, path)), dict(file=f, key=path, format=fmt), "result" if "results" in f else "data")


def JX(f, paths, fn, fmt, formula):
    """A value computed from values of one committed JSON (formula recorded)."""
    return _rec(fmt.format(fn(*[_get(f, p) for p in paths])), dict(file=f, keys=list(paths), formula=formula, format=fmt),
                "result" if "results" in f else "data")


def _const(path, name):
    tree = ast.parse((ROOT / path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                names = [t.id] if isinstance(t, ast.Name) else [e.id for e in getattr(t, "elts", []) if isinstance(e, ast.Name)]
                if name in names:
                    val = node.value
                    if isinstance(val, ast.Call) and getattr(val.func, "id", None) == "dict" and not val.args:
                        v = {k.arg: ast.literal_eval(k.value) for k in val.keywords}     # dict(key=literal, ...)
                        return v
                    v = ast.literal_eval(val)
                    return v[names.index(name)] if len(names) > 1 else v
    raise SystemExit(f"{path}: no constant {name}")


def C(path, name, fmt="{}", fn=None, formula=None):
    """A code constant, parsed from the source (optionally transformed; the formula is recorded)."""
    v = _const(path, name)
    v = fn(v) if fn else v
    return _rec(fmt.format(v), dict(code=f"{path}:{name}", format=fmt, **({"formula": formula} if formula else {})),
                "constant")


def P(path, section, pattern, group=1):
    """A number from the text of a pre-registration (regex; the match is recorded)."""
    m = re.search(pattern, (ROOT / path).read_text())
    if not m:
        raise SystemExit(f"{path}: {pattern!r} not found")
    return _rec(m.group(group), dict(file=path, section=section, match=m.group(0)), "design")


def T(path, text, kind="reference"):
    """Text copied from a repository file; it must occur there verbatim."""
    if text not in (ROOT / path).read_text():
        raise SystemExit(f"{path}: text not found: {text[:80]!r}")
    return _rec(text, dict(file=path, verbatim=True), kind)


def X(text, source, kind="reference"):
    """Text confirmed outside the repository (Crossref); the source is recorded."""
    return _rec(text, source, kind)


def doc(path):
    """A command line of a script's docstring, copied verbatim."""
    src = (ROOT / path).read_text()
    ds = ast.get_docstring(ast.parse(src), clean=False)

    def f(line):
        if line not in ds:
            raise SystemExit(f"{path}: command not in the docstring: {line}")
        return _rec(line, dict(file=path, docstring=True), "command")
    return f


def anchor(h):
    return re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-")


# ── shared values ─────────────────────────────────────────────────────────────────────────────────────────────────
def arms():
    return list(_load(REPORT)["runs"])


def seeds_n(arm):
    return J(REPORT, f"summary/{arm}/r2/n")


def fw_version():
    return T("scripts/viz_anatomy.py", "FlyWire v783", "name")


def banc_version():
    m = re.search(r"materialisation (v\d+)", (ROOT / "scripts/viz_anatomy.py").read_text())
    return _rec(m.group(1), dict(file="scripts/viz_anatomy.py", match=m.group(0)), "name")


def val_range():
    return (JX(SPLITS, ["val"], min, "{}", "min of val"), JX(SPLITS, ["val"], max, "{}", "max of val"))


def test_range():
    return (JX(SPLITS, ["test"], min, "{}", "min of test"), JX(SPLITS, ["test"], max, "{}", "max of test"))


def lateral():
    lat = _const("scripts/f5_train.py", "LATERAL")
    plus = ", ".join(m for m, s in lat.items() if s > 0)
    minus = ", ".join(m for m, s in lat.items() if s < 0)
    for txt, sign in ((plus, "+1"), (minus, "-1")):
        _rec(txt, dict(code="scripts/f5_train.py:LATERAL", formula=f"the types with sign {sign}"), "name")
    return plus, minus


def _np_std(v):
    import numpy as _np
    return float(_np.std(v))


def img(src, alt):
    return f'<img src="{src}" width="100%" alt="{alt.replace("`", "")}">'


# ── What the fly sees (Figure 7) ──────────────────────────────────────────────────────────────────────────────────
def _ic_sel(rows, v):
    """The train and validation rows of input_checks.csv at drum speed v (test rows carry no values)."""
    sel = [r for r in rows if r["split"] in ("train", "val") and int(r["speed_deg_s"]) == v]
    if not sel:
        raise SystemExit(f"{ICSV}: no train / val rows at {v} deg/s")
    return sel


def _ic_vals(rows, v, col):
    return [float(r[col]) for r in _ic_sel(rows, v)]


def IC(v, what, col=None, fmt="{:.3f}"):
    """A statistic of the train + val rows of input_checks.csv at speed v; every SD is the sample SD (ddof = 1)."""
    fns = {"n": (lambda rows: len(_ic_sel(rows, v)), f"train + val seed groups at {v} deg/s (count of rows)"),
           "mean": (lambda rows: statistics.mean(_ic_vals(rows, v, col)),
                    f"mean of {col} over the train + val rows at {v} deg/s"),
           "sd": (lambda rows: statistics.stdev(_ic_vals(rows, v, col)),
                  f"sample SD (ddof = 1) of {col} over the train + val rows at {v} deg/s"),
           "min": (lambda rows: min(_ic_vals(rows, v, col)), f"min of {col} over the train + val rows at {v} deg/s"),
           "max": (lambda rows: max(_ic_vals(rows, v, col)), f"max of {col} over the train + val rows at {v} deg/s")}
    fn, formula = fns[what]
    return JX(ICSV, ["rows"], fn, "{}" if what == "n" else fmt, formula)


def what_the_fly_sees(fw):
    """The body of the section "What the fly sees": text, Figure 7 and its two tables."""
    vi, rows = _load(VI), _load(ICSV)["rows"]
    rec = _load(RECORDS)["completion"]
    subtypes = vi["subtypes"]
    speeds = _const("scripts/f5_stageA_records.py", "SPEEDS")
    held = _const("train/f5_stageA.py", "HELD_OUT")
    gs = _const("scripts/readme_visual_input.py", "GATE_SPEEDS")
    used = [r for r in rows if r["split"] in ("train", "val")]
    # the statements below, checked against the data before they are printed
    if sorted({int(r["speed_deg_s"]) for r in used}) != sorted(speeds):
        raise SystemExit("input_checks.csv: the train / val speeds differ from SPEEDS")
    if not all(float(r["A_in_rot_left"]) > 0 > float(r["A_in_rot_right"]) for r in used):
        raise SystemExit("input_checks.csv: a train / val group without rot_left > 0 > rot_right")
    for col, sgn in (("A_in_rot_left", 1), ("A_in_rot_right", -1)):
        m = [sgn * statistics.mean(_ic_vals(rows, v, col)) for v in sorted(speeds)]
        if not all(a < b for a, b in zip(m, m[1:])):
            raise SystemExit(f"input_checks.csv: |mean {col}| does not grow with speed")
    if not (rec["input_checks_passed"] == rec["input_checks_total"] == len(rows)
            and rec["input_checks"]["test"]["values_kept"] == 0
            and all(r["A_in_rot_left"] == r["A_in_rot_right"] == "" for r in rows if r["split"] == "test")):
        raise SystemExit("not every input check passed, or test values were kept; the README text differs")
    for k in ("rot_left", "rot_right"):
        for v in gs:
            a = vi["gate"]["A_in_T4aT5a"][f"p0_v{v:03d}"][k]["recomputed"]
            if (a > 0) != (k == "rot_left"):
                raise SystemExit(f"gate A_in of {k} at {v} deg/s has the other sign; the README text differs")
    imgs = vi["eye_images"]["images"]
    if not (imgs[0]["t_s"] is None and imgs[1]["t_s"] is not None):
        raise SystemExit("the first two eye images are not the grey lead-in and the first stimulus time")
    if vi["heatmap_vmax_hz"]["value"] != vi["heatmap_vmax_hz"]["r_clip_hz"]:
        raise SystemExit("the heat-map colour limit is not the boundary-layer clip; the caption differs")

    v1 = C("scripts/readme_visual_input.py", "GATE_SPEEDS", fn=lambda g: g[0], formula="GATE_SPEEDS[0]")
    v2 = C("scripts/readme_visual_input.py", "GATE_SPEEDS", fn=lambda g: g[1], formula="GATE_SPEEDS[1]")
    h30 = C("train/f5_stageA.py", "HELD_OUT", fn=lambda h: h["secondary"], formula="HELD_OUT['secondary']")
    h90 = C("train/f5_stageA.py", "HELD_OUT", fn=lambda h: h["primary"], formula="HELD_OUT['primary']")
    k1 = f"rot_left_v{gs[0]:03d}_p0"
    lead = {e: JX(VI, ["lead_in_mean_hz/values"],
                  lambda d, e=e: _same([x[e] for x in d.values()]), "{:.3f}",
                  f"lead-in mean of eye {e}, equal for every key") for e in ("L", "R")}
    lo = JX(VI, [f"mean_rate_hz/{k1}"], lambda d: min(min(x.values()) for x in d.values()), "{:.1f}",
            "min over eyes and subtypes")
    hi = JX(VI, [f"mean_rate_hz/{k1}"], lambda d: max(max(x.values()) for x in d.values()), "{:.1f}",
            "max over eyes and subtypes")
    gate = {(k, v): J(VI, f"gate/A_in_T4aT5a/p0_v{v:03d}/{k}/recomputed", "{:.3f}")
            for k in ("rot_left", "rot_right") for v in gs}
    sym = JX(VI, ["asymmetry/values"], _max_sum, "{:.3f}",
             "largest |asymmetry(rot_left) + asymmetry(rot_right)| over subtypes and speeds")
    big = JX(VI, ["asymmetry/values"], lambda d: max(abs(x) for k in d.values() for x in k.values()), "{:.2f}",
             "largest |asymmetry| over subtypes and keys")
    sd_max = JX(ICSV, ["rows"], lambda r: max(statistics.stdev(_ic_vals(r, v, c)) for v in speeds
                                              for c in ("A_in_rot_left", "A_in_rot_right")), "{:.3f}",
                "largest sample SD (ddof = 1) of A_in over speeds and directions, train + val rows")
    n_same = JX(ICSV, ["rows"], lambda r: _same([len(_ic_sel(r, v)) for v in speeds]), "{}",
                "train + val seed groups per speed (equal at every speed)")
    s_lo, s_hi = min(speeds), max(speeds)
    m_lo, m_hi = IC(s_lo, "mean", "A_in_rot_left"), IC(s_hi, "mean", "A_in_rot_left")
    pv = "scripts/f5_stageA_preview.py"
    stim_seed = P(pv, "docstring", r"training data seed (\d+) \(phase")
    stim_phase = P(pv, "docstring", r"\(phase (\d+); never")
    pct = JX(VI, ["heatmap_vmax_hz/fraction_at_clip"], lambda f: 100 * f, "{:.1f}", "fraction x 100 (not rounded)")
    L = [f"The drum is rendered for the two FlyGym eyes and passed through FlyVis, whose boundary layer drives the "
          f"{fw} brain. T4 and T5 are the direction-selective motion detectors of the fly optic lobe, T4 for ON "
          f"(brightening) and T5 for OFF (darkening) edges, each in four subtypes, a to d; FlyVis models them "
          f"(Lappalainen et al., see the references), and the boundary layer carries their rates per eye. The "
          f"statements below are measurements; no preferred direction is attributed to any subtype.", "",
          f"In the grey lead-in the mean T4/T5 rate is {lead['L']} Hz in the left eye and {lead['R']} Hz in the "
          f"right; during leftward rotation at {v1} deg/s the per-subtype, per-eye means over the stimulus lie between {lo} and "
          f"{hi} Hz. With the input asymmetry `A_in = (L - R) / (L + R + 1)` of the mean left-eye and right-eye rates, "
          f"leftward rotation gives a positive A_in of T4a + T5a (left eye higher: {gate[('rot_left', gs[0])]} at {v1} "
          f"deg/s, {gate[('rot_left', gs[1])]} at {v2} deg/s) and rightward rotation a negative one "
          f"({gate[('rot_right', gs[0])]} and {gate[('rot_right', gs[1])]}). Per subtype, the asymmetry under rightward "
          f"rotation is close to the negative of that under leftward rotation: their sum is at most {sym} in absolute "
          f"value over the subtypes and both speeds, against single asymmetries of up to {big}.", "",
          f"Over every training and validation seed group (they differ in grating phase), A_in of T4a + T5a is "
          f"positive for rot_left and negative for rot_right in every group, and its size grows with drum speed: the "
          f"rot_left mean rises from {m_lo} at {C('scripts/f5_stageA_records.py', 'SPEEDS', fn=min, formula='min of SPEEDS')} "
          f"deg/s to {m_hi} at {C('scripts/f5_stageA_records.py', 'SPEEDS', fn=max, formula='max of SPEEDS')} deg/s. "
          f"The grating phase hardly changes it: the largest SD over speeds and directions is {sd_max}. All "
          f"{J(RECORDS, 'completion/input_checks_total')} input checks passed (training, validation and test); for the test split only the pass was kept, not the values.",
          "",
          img("docs/figures/fig7_visual_input.png",
              "Four panels: for each eye, small grey and striped eye images above a space-time plot of luminance with "
              "slanted stripes; heat maps of motion-detector rates grouped by subtype over time for both eyes and both "
              "rotation directions; dots of the per-subtype eye asymmetry; and the input-check asymmetry per seed "
              "group against drum speed."), "",
          f"**Figure 7 | What the fly sees and how its T4/T5 motion detectors respond.** a, Raw images of the left and "
          f"right FlyGym eye (the grey lead-in and the first stimulus time, t = "
          f"{J(VI, 'eye_images/images/1/t_s', '{:g}')} s) and, per eye, the luminance of the middle image row against "
          f"time from drum onset and azimuth (positive to the left of the fly; the left plot's axis label applies to "
          f"both); the drum turns as one body, so the stripes have the same slope in both eyes. b, T4/T5 boundary "
          f"rates of the left and right eye during leftward and rightward rotation at {v1} deg/s; rows are neurons "
          f"grouped by subtype, so each pixel row averages several neurons; the colour scale runs from 0 to the "
          f"boundary-layer clip of {J(VI, 'heatmap_vmax_hz/r_clip_hz', '{:g}')} Hz, which {pct} % of the plotted "
          f"rates reach. c, Per-subtype A_in at {v1} (circles) and {v2} deg/s (squares); blue, rot_left; vermillion, rot_right. d, A_in of T4a + T5a of the input check per seed "
          f"group ({n_same} training and validation seed groups per speed); the held-out speeds {h30} and {h90} deg/s on "
          f"the grey band; dashed lines at ±{C('scripts/f3b1_common.py', 'INPUT_CHECK_MIN', '{:g}')}, the input-check "
          f"minimum. a-c, training data seed {stim_seed}, grating phase {stim_phase} (the renders were re-made and shown "
          f"identical to the originals); d, every training and validation seed group of `docs/figures/input_checks.csv`. "
          f"The original previews are Figures S8-S11 in [docs/FIGURES.md](docs/FIGURES.md).", ""]
    hdr = [f"rot_left {v1}", f"rot_right {v1}", f"rot_left {v2}", f"rot_right {v2}"]
    keys = [f"{k}_v{v:03d}_p0" for v in gs for k in ("rot_left", "rot_right")]
    L += ["<details><summary>T4/T5 per subtype</summary>", "",
          f"| subtype | neurons, left / right eye | mean rate, rot_left {v1} deg/s, left / right eye (Hz) | "
          + " | ".join(f"A_in, {h}" for h in hdr) + " |",
          "|---|---:|---:|" + "---:|" * len(hdr)]
    for t in subtypes:
        L.append(f"| {t} | {J(VI, f'neurons/L/per_subtype/{t}', '{:,}')} / {J(VI, f'neurons/R/per_subtype/{t}', '{:,}')} "
                 f"| {J(VI, f'mean_rate_hz/{k1}/L/{t}', '{:.1f}')} / {J(VI, f'mean_rate_hz/{k1}/R/{t}', '{:.1f}')} | "
                 + " | ".join(J(VI, f"asymmetry/values/{k}/{t}", "{:+.2f}") for k in keys) + " |")
    L += [f"| all | {J(VI, 'neurons/L/t4t5_total', '{:,}')} / {J(VI, 'neurons/R/t4t5_total', '{:,}')} | | "
          + " | " * (len(hdr) - 1) + " |", "",
          "Mean rates and A_in over the whole stimulus (the input-check window of `scripts/f3b1_common.py`), training "
          f"data seed {stim_seed}, grating phase {stim_phase}; `docs/figures/visual_input.json`.", "", "</details>", ""]
    L += ["<details><summary>The input check over the training data</summary>", "",
          "| speed (deg/s) | held out? | seed groups (train + val) | A_in rot_left, mean ± SD (min) | "
          "A_in rot_right, mean ± SD (max) |", "|---:|---|---:|---:|---:|"]
    for i, v in enumerate(speeds):
        L.append(f"| {C('scripts/f5_stageA_records.py', 'SPEEDS', fn=lambda s, i=i: s[i], formula=f'SPEEDS[{i}]')} | "
                 f"{'yes' if v in held.values() else 'no'} | {IC(v, 'n')} | "
                 f"{IC(v, 'mean', 'A_in_rot_left')} ± {IC(v, 'sd', 'A_in_rot_left')} ({IC(v, 'min', 'A_in_rot_left')}) | "
                 f"{IC(v, 'mean', 'A_in_rot_right')} ± {IC(v, 'sd', 'A_in_rot_right')} "
                 f"({IC(v, 'max', 'A_in_rot_right')}) |")
    L += ["", "A_in of T4a + T5a over the whole stimulus, one value per seed group and direction (prereg/09 input "
          "check: it passes when rot_left and rot_right have opposite signs and both are at least the input-check "
          "minimum in size); SD, the sample SD over seed groups (`ddof = 1`); min, the smallest rot_left value; max, "
          "the largest (least negative) rot_right value. `docs/figures/input_checks.csv`; the test rows carry only the "
          "pass.", "", "</details>", ""]
    return L


# ── Training tables ───────────────────────────────────────────────────────────────────────────────────────────────
def _held_out_trials(n_test_seeds):
    """Rotation trials of the held-out speeds per test evaluation: every rotation stimulus of TRAIN_CONDITIONS at every
    speed of HELD_OUT, on every test seed (the groups of A.TEST_GROUPS, as scripts/f5_stageA_report.py checks)."""
    cond, held = _const("train/f5_stageA.py", "TRAIN_CONDITIONS"), _const("train/f5_stageA.py", "HELD_OUT")
    return len({s for s, _ in cond if s.startswith("rot")}) * len(held) * n_test_seeds


def _n_test_trials(runs):
    return _same([r["n_test_trials"] for arm in runs.values() for r in arm.values()])


def training_data(sp1, sp2, held30, held90, v_lo, v_hi, t_lo, t_hi):
    """The training-data table: data seeds and grating phase (prereg/14 section 7), records per split (the records
    plan), the trials stage A uses (the runner design and the test report). The counts are checked against the split
    file and the conditions, and the statements of the held-out-speed note against the code, before printing."""
    sp, plan = _load(SPLITS), _load(RECORDS)["plan"]
    rn, rep = _load(RUNNER)["design"], _load(REPORT)
    cond = _const("train/f5_stageA.py", "TRAIN_CONDITIONS")
    speeds = _const("scripts/f5_stageA_records.py", "SPEEDS")
    held = _const("train/f5_stageA.py", "HELD_OUT")
    n_cond = P(P14, "7", r"static, grey \((\d+)\)\.")
    ph0 = P(P14, "7", r"\*\*Grating phase\.\*\* (\d+) for data seeds")
    ph_lo = P(P14, "7", r"phase\.\*\* \d+ for data seeds (\d+)-\d+;")
    ph_hi = P(P14, "7", r"phase\.\*\* \d+ for data seeds \d+-(\d+);")
    ph_next = P(P14, "7", r"`default_rng\(seed\)` for seeds (\d+)-\d+\.")
    m = re.search(r"^PHASE0_SEEDS = range\((\d+), (\d+)\)$", (ROOT / "scripts/f5_stageA_records.py").read_text(), re.M)
    if not m or (int(m.group(1)), int(m.group(2)) - 1) != (int(ph_lo), int(ph_hi)) or int(ph_next) != int(ph_hi) + 1:
        raise SystemExit("the phase-0 seeds of scripts/f5_stageA_records.py differ from prereg/14 section 7")
    if not (min(sp["train"]) == int(ph_lo) and int(ph_next) <= max(sp["train"]) < min(sp["val"])):
        raise SystemExit("the phase-0 seeds are not the first training seeds")
    if sorted(speeds) != sorted({v for _, v in cond if v} | set(held.values())):
        raise SystemExit("the record speeds are not the training plus the held-out speeds")
    for s in ("train", "val", "test"):
        if plan["per_split"][s] != len(sp[s]) * int(n_cond):
            raise SystemExit(f"{s}: the records are not {n_cond} conditions per data seed")
    if sum(plan["per_split"].values()) != plan["n_records"]:
        raise SystemExit("the records per split do not add up to the records")
    n_tt, n_ho = _n_test_trials(rep["runs"]), _held_out_trials(rep["test_seeds"])
    if (rn["n_train_trials"], rn["n_val_trials"], n_tt) != tuple(len(sp[s]) * len(cond) for s in ("train", "val", "test")):
        raise SystemExit("the stage-A trials are not one per data seed and training condition")
    if rep["test_seeds"] != len(sp["test"]) or n_tt + n_ho != plan["per_split"]["test"]:
        raise SystemExit("a test evaluation does not use every test record once")
    for code_file, needle in (
            ("train/f5_stageA.py", 'raise SealedSplitError("the held-out drum speeds (90, 30 deg/s) are read only by '
                                   'the final evaluation")'),
            ("scripts/f5_stageA_posthoc.py", "raise A.SealedSplitError(f\"data seed {seed} is a sealed test seed; the "
                                             "post-hoc reader never reads it\")"),
            ("scripts/f5_stageA_posthoc.py", 'raise PosthocRefusal(f"data seed {seed} is not a validation seed '
                                             '160-199")'),
            ("scripts/f5_stageA_posthoc.py", 'data_cache[key] = A.load_trials("val", setup, bridge, A.TRAIN_CONDITIONS)'),
            ("scripts/f5_stageA_records_report.py", '"""File integrity and record format of one training-seed record; no '
                                                    'statistic of its content."""')):
        if needle not in (ROOT / code_file).read_text():
            raise SystemExit(f"{code_file} no longer contains {needle!r}; the held-out-speed note differs")
    tr_lo, tr_hi = JX(SPLITS, ["train"], min, "{}", "min of train"), JX(SPLITS, ["train"], max, "{}", "max of train")
    s = [C("scripts/f5_stageA_records.py", "SPEEDS", fn=lambda x, i=i: x[i], formula=f"SPEEDS[{i}]")
         for i in range(len(speeds))]
    uniform = "uniform over one wavelength"
    tt = JX(REPORT, ["runs"], _n_test_trials, "{}", "n_test_trials, equal for every run")
    ho = JX(REPORT, ["test_seeds"], _held_out_trials, "{}", "rotation stimuli of TRAIN_CONDITIONS x len(HELD_OUT) x "
                                                                "test_seeds")
    total = JX(REPORT, ["runs", "test_seeds"], lambda r, n: _n_test_trials(r) + _held_out_trials(n), "{}",
               "n_test_trials (equal for every run) + rotation stimuli of TRAIN_CONDITIONS x len(HELD_OUT) x test_seeds")
    return ["The training data (prereg/14 section 7):", "",
            "| split | data seeds | grating phase | records: drum speeds and conditions | trials stage A uses: the "
            "training set; one validation evaluation; one test evaluation |", "|---|---|---|---|---:|",
            f"| training | {tr_lo}-{tr_hi} | {ph0} for seeds {ph_lo}-{ph_hi}; {uniform} (`default_rng(seed)`) for "
            f"{ph_next}-{tr_hi} | {J(RECORDS, 'plan/per_split/train', '{:,}')}: rot_left / rot_right at {s[0]}, {s[1]}, "
            f"{s[2]} and {s[3]} deg/s, static, grey ({n_cond} per data seed) | "
            f"{J(RUNNER, 'design/n_train_trials', '{:,}')} |",
            f"| validation | {v_lo}-{v_hi} | {uniform} | {J(RECORDS, 'plan/per_split/val', '{:,}')}, the same "
            f"{n_cond} conditions | {J(RUNNER, 'design/n_val_trials')} |",
            f"| test | {t_lo}-{t_hi} | {uniform} | {J(RECORDS, 'plan/per_split/test', '{:,}')}, the same {n_cond} "
            f"conditions | {total}: {tt} at the training speeds, {ho} held-out rotation trials |", "",
            f"- **Trials:** one per data seed and condition. Training and validation use rot_left / rot_right at {sp1} "
            f"and {sp2} deg/s, static and grey; a test evaluation adds rot_left / rot_right at the held-out speeds "
            f"{held30} and {held90} deg/s. The {J(RUNNER, 'design/n_train_trials', '{:,}')} trials of the training set "
            f"are not the trials presented: training draws its batches from them, "
            f"{JX(RUNNER, ['design/steps', 'design/batch'], lambda a, b: a * b, '{:,}', 'steps x batch')} trials per "
            f"run (How it was trained, below).",
            "- **Test:** each run was evaluated on the test split once, on the CPU, from its selected checkpoint, after "
            "training and learning-rate selection; no selection step used it (see "
            "[Methods and safeguards](#methods-and-safeguards)).",
            f"- **Held-out speeds:** the {held30} and {held90} deg/s records of the training and validation seeds were "
            f"read by neither training nor selection; the training code refuses them (`SealedSplitError` in "
            f"`load_trials` of `train/f5_stageA.py`). The post-hoc analyses (prereg/21, Figure 4) later read the "
            f"validation records at those speeds through their own reader in `scripts/f5_stageA_posthoc.py`, which "
            f"bypasses `load_trials` for the held-out speeds only, accepts only validation seeds and refuses every test "
            f"seed. Apart from that, a few training-seed records at those speeds were opened for file and format checks "
            f"only (`scripts/f5_stageA_records_report.py`, no statistic of their content), and the SHA-256 of every "
            f"record is recomputed from its bytes (`scripts/verify_report.py`).",
            "- **Input:** what the fly's eyes see in these records and how its T4/T5 motion detectors respond is shown "
            "in [What the fly sees](#what-the-fly-sees).", ""]


def _nz4(path):
    """A mean at 4 decimals; one that would print as -0.0000 goes through the no-negative-zero formula of the Key
    numbers table."""
    if "{:.4f}".format(_get(VALID, path)) != "-0.0000":
        return J(VALID, path, "{:.4f}")
    return JX(VALID, [path], lambda v: round(v, 4) + 0.0, "{:.4f}", "mean rounded to 4 decimals (+ 0.0: no negative zero)")


def _steps(arm):
    return [r["selected_step"] for r in _load(VALID)["runs"][arm].values()]


def lr_decimals(values):
    """The fewest decimals, at least LR_MIN_DECIMALS, at which the printed values are all distinct."""
    for d in range(LR_MIN_DECIMALS, 17):
        if len({f"{v:.{d}f}" for v in values}) == len(values):
            return d
    raise SystemExit(f"values equal at every printable precision: {values}")


def validation_tables(vs, v_lo, v_hi, seed0):
    """Validation at the selected checkpoint, one row per arm, and the learning-rate grid of training seed 0 (both from
    docs/results/f5_stageA_validation.json; its SDs are sample SDs, ddof = 1)."""
    if vs["sd"] != "sample sd over training seeds, ddof = 1" or vs["split"] != "validation" or vs["test_evaluated"]:
        raise SystemExit("the validation report is not the validation split with sample SDs")
    acc = {a: s["best_sign_accuracy"]["mean"] for a, s in vs["summary"].items()}
    if min(v for a, v in acc.items() if a != "bridge_off") < 0.99:
        raise SystemExit("an arm other than bridge_off is not near 1 in turn-sign accuracy; the README text differs")
    L = ["Validation at the selected checkpoint (validation split, descriptive; not the test):", "",
         "| arm | selected learning rate | runs | best validation loss, mean ± SD | best R2, mean ± SD | best turn-sign "
         "accuracy, mean | selected step, median (min-max) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for a, s in vs["summary"].items():
        if s["n_seeds"] != len(vs["runs"][a]) or (a == "untrained") != (a not in vs["selected_lr"]):
            raise SystemExit(f"{a}: runs, seeds or selected rate differ")
        if a == "untrained" and (_steps(a) != [0] or any(s[m]["sd"] is not None for m in ("best_loss", "best_r2"))):
            raise SystemExit("untrained is not one evaluation at initialisation")
        ms = [f"{_nz4(f'summary/{a}/{m}/mean')} ± " + ("-" if s[m]["sd"] is None else
                                                        J(VALID, f"summary/{a}/{m}/sd", "{:.4f}"))
              for m in ("best_loss", "best_r2")]
        lr = "-" if a == "untrained" else J(VALID, f"selected_lr/{a}", "{:g}")
        step = "-" if a == "untrained" else (
            f"{JX(VALID, [f'runs/{a}'], lambda r: statistics.median(x['selected_step'] for x in r.values()), '{:,.0f}', 'median of selected_step over the runs')} "
            f"({JX(VALID, [f'runs/{a}'], lambda r: min(x['selected_step'] for x in r.values()), '{:,}', 'min of selected_step over the runs')}-"
            f"{JX(VALID, [f'runs/{a}'], lambda r: max(x['selected_step'] for x in r.values()), '{:,}', 'max of selected_step over the runs')})")
        L.append(f"| `{a}` | {lr} | {J(VALID, f'summary/{a}/n_seeds')} | {ms[0]} | {ms[1]} | "
                 f"{J(VALID, f'summary/{a}/best_sign_accuracy/mean', '{:.3f}')} | {step} |")
    bo, un = (J(VALID, f"summary/{a}/best_sign_accuracy/mean", "{:.3f}") for a in ("bridge_off", "untrained"))
    L += ["", f"Validation data seeds {v_lo}-{v_hi}, {J(RUNNER, 'design/n_val_trials')} trials per evaluation. Best: at "
              f"the selected checkpoint, the evaluation with the lowest validation loss. SD: {J(VALID, 'sd')}. "
              f"`untrained` is one evaluation at initialisation (no learning rate, no SD). Turn-sign accuracy is at or "
              f"near ceiling in every arm but `bridge_off` ({bo}), and `untrained` already reaches {un} at "
              f"initialisation, so it separates the arms little (prereg/20 limitation b, see [Limitations](#limitations)); loss and R2 are "
              f"the informative columns.", ""]
    grid = _load(RUNNER)["design"]["lr_grid"]
    dmin = C("scripts/readme_text.py", "LR_MIN_DECIMALS")
    L += [f"<details><summary>Learning-rate grid (training seed {seed0})</summary>", "",
          "| arm | " + " | ".join(f"lr {J(RUNNER, f'design/lr_grid/{i}', '{:g}')}" for i in range(len(grid))) + " |",
          "|---|" + "---:|" * len(grid)]
    for a, g in vs["lr_grid"].items():
        losses = [x["final_val_loss"] for x in g]
        if [x["lr"] for x in g] != grid or min(g, key=lambda x: (x["final_val_loss"], x["lr"]))["lr"] != vs["selected_lr"][a]:
            raise SystemExit(f"{a}: the learning-rate grid or its selection differs from the rule")
        d = lr_decimals(losses)
        cells = [J(VALID, f"lr_grid/{a}/{i}/final_val_loss", f"{{:.{d}f}}") for i in range(len(g))]
        L.append(f"| `{a}` | " + " | ".join(f"**{c}**" if x["lr"] == vs["selected_lr"][a] else c
                                             for c, x in zip(cells, g)) + " |")
    if [a for a, g in vs["lr_grid"].items() if len({f"{x['final_val_loss']:.{LR_MIN_DECIMALS}f}" for x in g}) < len(g)] \
            != ["bridge_off"]:
        raise SystemExit("an arm other than bridge_off would tie at the minimum decimals; the README text differs")
    rule = T("docs/results/f5_stageA_validation.md", "lowest final validation loss at training seed 0 (prereg/14 "
             "section 6); a tie goes to the smaller rate.", "text")
    L += ["", f"Final validation loss (step {J(RUNNER, 'design/steps', '{:,}')}) of the seed-{seed0} run of each arm "
              f"at each rate; bold, the selected rate. Selection rule (`docs/results/f5_stageA_validation.md`): {rule} "
              f"Each arm is printed with the fewest decimals, at least {dmin}, that keep its three values apart; "
              f"`bridge_off` therefore has more: at {dmin} its values would look tied, which they are not. Every run at "
              f"the selected rate, seed by seed: "
              f"[docs/results/f5_stageA_validation.md](docs/results/f5_stageA_validation.md#2-runs-at-the-selected-rate).",
          "", "</details>", ""]
    return L


def _same(values):
    """The common value of a list whose values are all equal."""
    if not values or any(x != values[0] for x in values):
        raise SystemExit(f"values differ: {values}")
    return values[0]


def _max_sum(asym):
    out = []
    for k, v in asym.items():
        if k.startswith("rot_left_"):
            w = asym["rot_right_" + k[len("rot_left_"):]]
            out += [abs(v[t] + w[t]) for t in v]
    return max(out)


# ── README ────────────────────────────────────────────────────────────────────────────────────────────────────────
def readme():
    CUR["file"] = "README.md"
    rep, ph = _load(REPORT), _load(POSTHOC)
    runs = rep["runs"]
    pr = rep["primary"]
    if pr["verdict"] != "The connectome helps in stage A" or pr["status"] != "final":
        raise SystemExit("the verdict in the report differs from the README text")
    if not all(x > 0 for x in pr["d"]):
        raise SystemExit("not every training seed has d > 0; the README text says every seed")
    best_train = max(arms(), key=lambda a: rep["summary"][a]["r2"]["mean"])
    above90 = [a for a in arms() if rep["summary"][a]["g90_r2"]["mean"] > rep["summary"]["true"]["g90_r2"]["mean"]]
    if above90 != ["shuffled", "mlp"]:
        raise SystemExit(f"arms above true at 90 deg/s: {above90}; the README text differs")
    best_30 = max(arms(), key=lambda a: rep["summary"][a]["g30_r2"]["mean"])
    if best_train != "mlp" or best_30 != "true":
        raise SystemExit(f"highest at the training speeds {best_train}, at 30 deg/s {best_30}; the README text differs")
    if ph["A"]["primary"]["label"] != "replicated on the validation seeds (post-hoc)":
        raise SystemExit("the post-hoc label differs from the README text")
    trained = [a for a in arms() if a not in ("bridge_off", "untrained")]
    plus, minus = lateral()
    v_lo, v_hi = val_range()
    t_lo, t_hi = test_range()
    held30, held90 = C("train/f5_stageA.py", "HELD_OUT", fn=lambda h: h["secondary"], formula="HELD_OUT['secondary']"), \
        C("train/f5_stageA.py", "HELD_OUT", fn=lambda h: h["primary"], formula="HELD_OUT['primary']")
    train_speeds = sorted({v for _, v in _const("train/f5_stageA.py", "TRAIN_CONDITIONS") if v})
    sp1 = C("train/f5_stageA.py", "TRAIN_CONDITIONS", fn=lambda c: sorted({v for _, v in c if v})[0],
            formula="lowest speed of TRAIN_CONDITIONS")
    sp2 = C("train/f5_stageA.py", "TRAIN_CONDITIONS", fn=lambda c: sorted({v for _, v in c if v})[1],
            formula="highest speed of TRAIN_CONDITIONS")
    assert train_speeds == [int(sp1), int(sp2)]
    fw, banc = fw_version(), banc_version()
    n_true = seeds_n("true")
    seed0 = P("prereg/14_f5_training.md", "6", r"final validation loss\s+at training seed (\d+) is used")
    r2_one = P(P14, "8.1", r"\| \*\*R2 \(primary A, B\)\*\* \| (\d) - SSE / SST")
    sq = P(P14, "6", r"deg\)\^(\d)\.")
    zero_off = P(P14, "5", r"every bridged DN clamped to (\d+)")
    zero_dn = P(P14, "2", r"every other BANC DN is clamped at (\d+)")
    zero_teacher = P(P14, "4", r"dA\* = (\d+) under grey")
    g_teacher = P(P21, "3.3", r"G = (\d+) is the teacher")
    crit_pct = P(P21, "4.4", r"two-sided (\d+) % critical value")
    ci_pct = JX(REPORT, ["primary/effect/level"], lambda v: round(100 * v), "{}", "level x 100")
    bin_pct = JX(REPORT, ["primary/guards/leakage/interval/level"], lambda v: round(100 * v), "{}", "level x 100")
    cc_nc = T("docs/figures/README.md", "CC BY-NC 4.0", "name")
    cc_by = T("docs/figures/README.md", "CC BY 4.0", "name")
    units = P(P19, "2", r"unchanged: \*\*(\d+) units\*\*")
    targets = P(P19, "2", r"units\*\*, (\d+) BANC targets")
    n_neu = P(P19, "1", r"\(([\d,]+) neurons, [\d,]+ edges\)")
    n_edge = P(P19, "1", r"\([\d,]+ neurons, ([\d,]+) edges\)")
    swaps = P("prereg/14_f5_training.md", "5", r"(\d+ x E) successful swaps")
    mlp_units = P(P19, "2", r"is the (\d+) units of the `true` bridge")
    n_types = JX(POSTHOC, ["B/targets/names"], lambda n: len(set(x.split("|")[0] for x in n)), "{}",
                 "number of MN types in B/targets/names")
    L = []
    headings = []

    def H(level, text):
        headings.append(text)
        L.extend([f"{'#' * level} {text}", ""])

    L += ["# synaptera-banc-brain", "",
          f"A simulation test of whether the real wiring of a BANC {banc} ventral nerve cord (VNC) subgraph, driven by "
          f"a {fw} brain model through a descending-neuron (DN) bridge, helps it learn a steering readout better than "
          "the same subgraph with degree-preserving shuffled wiring.", "",
          "Ömer Faruk Topal · [github.com/omeruk](https://github.com/omeruk)", ""]
    toc = ["Result in brief", "The stage-A circuit", "Activity on the circuit", "Post-hoc analyses",
           "How we got here (F0-F3)", "Key numbers",
           "How it works", "What the fly sees", "Joining two flies: cell types and coordinates", "Training", "Methods and safeguards", "Limitations", "Reproduce", "Repository layout",
           "Data, credits and licences", "Authorship", "How to cite"]
    L += ["**Contents:** " + " · ".join(f"[{t}](#{anchor(t)})" for t in toc), ""]
    speed = C("scripts/viz_video.py", "SPEED")
    L += [img("docs/figures/stageA_circuit_direction.gif",
              "Animation of the stage-A circuit seen from behind: brain and nerve-cord neurons light up orange or blue "
              "by their preference for rightward or leftward drum rotation, above a trace of the turn command for both "
              "directions."), "",
          f"*Direction preference of the stage-A circuit: per neuron, the trial-mean activity for rightward minus "
          f"leftward drum rotation at {speed} deg/s (orange prefers rightward, blue leftward), with the turn command dA "
          f"for both directions below; validation data seeds {v_lo}-{v_hi}, slowed "
          f"{C('scripts/viz_video.py', 'SLOW')}x, open loop (the fly does not turn).*", ""]

    H(2, "Result in brief")
    e, tst, holm = pr["effect"], pr["test"], pr["holm"]
    pe, ptest = ph["A"]["primary"]["effect"], ph["A"]["primary"]["test"]
    L += [f"The pre-registered stage-A verdict is \"{J(REPORT, 'primary/verdict')}\" ({J(REPORT, 'primary/status')}). "
          f"In plain words: on the sealed test split, the VNC subgraph trained with its real wiring (`true`) reached a "
          f"higher test R2 at the training speeds than the same subgraph with degree-preserving shuffled wiring "
          f"(`shuffled`), in all {JX(REPORT, ['primary/d'], len, '{}', 'len(primary/d)')} training seeds. R2 is "
          f"{r2_one} - SSE / SST of the window-mean turn command dA against the teacher dA*; d is the per-seed difference "
          f"R2(`true`) - R2(`shuffled`). Mean d was {J(REPORT, 'primary/effect/mean', '{:.4f}')} ({ci_pct} % bootstrap "
          f"interval {J(REPORT, 'primary/effect/lo', '{:.4f}')} to {J(REPORT, 'primary/effect/hi', '{:.4f}')}), "
          f"one-sided exact sign-flip p = {J(REPORT, 'primary/test/p_greater', '{:.4f}')} "
          f"({J(REPORT, 'primary/test/count_greater')}/{J(REPORT, 'primary/test/n_patterns')}), Holm-adjusted p = "
          f"{J(REPORT, 'primary/holm/adjusted/A', '{:.4f}')} (stages B and C were not run and count as p = "
          f"{J(REPORT, 'primary/holm/p/B', '{:g}')}).", "",
          f"Context. An unconstrained two-layer network (`mlp`) had a higher test R2 than `true` at the training "
          f"speeds. At the held-out {held90} deg/s, `shuffled` and `mlp` had a higher test R2 than `true`. "
          f"At the held-out {held30} deg/s, `true` had the highest test R2 of all "
          f"{JX(REPORT, ['runs'], len, '{}', 'number of arms in runs')} arms. On the validation data seeds the "
          f"{held30} deg/s advantage was found again (post-hoc, exploratory: mean d "
          f"{J(POSTHOC, 'A/primary/effect/mean', '{:.3f}')}, one-sided p = "
          f"{J(POSTHOC, 'A/primary/test/count_greater')}/{J(POSTHOC, 'A/primary/test/n_patterns')}); it used the same "
          "trained networks on new trials, so it is not an independent replication of training.", "",
          img("docs/figures/fig1_stageA_result.png",
              "Three scatter panels of test R2: true above shuffled for every training seed; the five trained arms "
              "with mlp highest; every arm per condition, where true is highest at the slowest held-out speed."), "",
          f"**Figure 1 | On the test split, `true` reached a higher R2 than `shuffled` at the training speeds; `mlp` "
          f"was higher still, and at {held30} deg/s `true` was highest.** a, Test R2 at the training speeds "
          f"({sp1} and {sp2} deg/s) of `true` and `shuffled` per training seed; grey lines join the two runs of a seed, "
          f"bars are arm means. b, Test R2 at the training speeds of the five trained comparison arms, one dot per "
          f"training seed (horizontal jitter only), bars are arm means; `bridge_off` and `untrained` lie far below "
          f"this axis (values in [Key numbers](#key-numbers)). c, Test R2 per condition: the training speeds, "
          f"{held90} deg/s and {held30} deg/s (both held out, grey bands); large markers are arm means, small dots "
          f"single runs; colours and marker shapes as in the legend, the same colours as Figure 4. Test split, data "
          f"seeds {t_lo}-{t_hi}: per run {J(REPORT, 'runs/true/t0/n_test_trials')} trials at the training speeds and "
          f"{JX(REPORT, ['test_seeds'], lambda n: 2 * n, '{}', '2 directions x test_seeds')} rotation trials per "
          f"held-out speed; {n_true} training seeds per trained arm, {seeds_n('bridge_off')} for `bridge_off`, "
          f"{seeds_n('untrained')} `untrained` run.", ""]

    H(2, "The stage-A circuit")
    L += [img("docs/figures/stageA_circuit_anatomy.png",
              "Four views of the fly brain and nerve cord in grey with the circuit drawn on top: FlyWire descending "
              "neurons in sky blue in the brain, BANC descending neurons in blue running down the neck, nerve-cord "
              "interneurons in orange and steering motor neurons in vermillion."), "",
          f"**Figure 2 | The stage-A circuit in BANC space, from the FlyWire DNs that drive the model to the steering "
          f"motor neurons it reads out.** a-d, Frontal (a), dorsal (b), lateral from the left (c) and posterior (d) "
          f"views at one scale. Sky blue, the {J(WARP, 'bridged_dns')} FlyWire descending neurons (DNs) that drive the "
          f"model; blue, the {J(POSTHOC, 'B/sources/n')} BANC DNs that receive their input; orange, "
          f"{C('scripts/viz_anatomy.py', 'N_INTERNEURONS')} VNC interneurons on the strongest two-hop paths to the "
          f"steering motor neurons; vermillion, the {J(POSTHOC, 'B/targets/n')} wing steering motor neurons (MNs) of "
          f"the readout; grey, brain and VNC neuropil. FlyWire DNs (brain) and BANC DNs (cord) of the same types meet "
          f"at the neck, where the model's DN bridge joins them. D/V, L/R, A/P: dorsal/ventral, left/right, "
          f"anterior/posterior. Scale bar, {C('scripts/viz_anatomy.py', 'SCALE_UM')} µm. Anatomy only; a demonstration "
          f"figure.", ""]

    H(2, "Activity on the circuit")
    n_val = JX(SPLITS, ["val"], len, "{}", "len(val)")
    L += [img("docs/figures/stageA_circuit_activity.png",
              "Dorsal and lateral views of the circuit with each neuron coloured blue to orange by its direction "
              "preference, a dot plot of left-minus-right motor-neuron states per type, and turn-command traces that "
              "follow the dashed teacher."), "",
          f"**Figure 3 | Direction preference of every drawn neuron of the trained circuit, and the turn command it "
          f"produced, on validation trials.** a, b, Dorsal (a) and lateral (b) views; each neuron is coloured by its "
          f"mean window activity for rightward minus leftward rotation (orange, rightward preferred; blue, leftward), "
          f"strongest on top. FlyWire DNs in Hz, from the untrained spiking reference brain; BANC DNs, VNC "
          f"interneurons and steering MNs in units of the nerve-cord rate model (`true`, training seed "
          f"{C('scripts/viz_activity_figure.py', 'TRAINING_SEED')}), the only part trained in stage A; the colour bars "
          f"give the ranges. c, Left-minus-right state per steering-MN type and direction (orange rightward, blue "
          f"leftward), mean ± s.d.; +/-, readout sign. d, Turn command dA (deg) over time from stimulus onset (ms) per "
          f"direction and speed (solid {sp2}, dotted {sp1} deg/s), mean ± s.d.; dashed, the teacher dA*; shading, the "
          f"analysis window. Validation split, data seeds {v_lo}-{v_hi} ({n_val} seeds) at {sp1} and {sp2} deg/s: "
          f"{JX(SPLITS, ['val'], lambda v: 4 * len(v), '{}', '2 directions x 2 speeds x len(val)')} rotation trials.",
          ""]

    H(2, "Post-hoc analyses")
    B = ph["B"]
    n_out = JX(POSTHOC, ["B/measures"], lambda m: sum(x["outside_shuffled_range"] for x in m.values()), "{}",
               "number of measures with outside_shuffled_range")
    n_meas = JX(POSTHOC, ["B/measures"], len, "{}", "len(B/measures)")
    s = B["correlations"]["strongest"]
    if B["correlations"]["n_at_or_above_crit"] != 0:
        raise SystemExit("a correlation reached the critical value; the Figure 4 caption says none did")
    L += ["These analyses were planned after the test result (prereg/21, committed before they were computed) and "
          "are exploratory; none of them enters a verdict.", "",
          img("docs/figures/posthoc_stageA_figure.png",
              "Four panels: gain ratio against drum speed per arm; validation R2 at the slowest held-out speed, true "
              "above shuffled for every seed; z-scores of graph measures of the true graph against the shuffled graphs; a scatter of "
              "one graph measure against the per-seed deficit."), "",
          f"**Figure 4 | Post-hoc, exploratory: on the validation data seeds `true` kept its advantage at {held30} "
          f"deg/s, and its graph lay outside the range of the shuffles on {n_out} of {n_meas} structure measures.** "
          f"a, Speed tuning: gain ratio G(v) = mean sign(dA*) dA / (K_P v) of the window-mean output against drum "
          f"speed, per arm; lines, means over the training seeds; small dots, single seeds; `untrained`, one run; "
          f"dashed line, the teacher (G = {g_teacher}); the held-out speeds are shaded. b, Validation R2 at {held30} deg/s, "
          f"`true` against `shuffled` per training seed (lines pair the seeds), with the primary post-hoc test in the "
          f"panel. c, Structure of the `true` VNC graph against the shuffled graphs: per measure S1-S18 "
          f"([definitions](docs/results/f5_stageA_posthoc.md)), z = (value - shuffled mean) / shuffled SD; grey, the "
          f"shuffled graphs; blue, `true` (filled where it lies outside the shuffled range); symmetric-log axis. d, The "
          f"strongest structure-deficit correlation, selected as the largest |rho| of "
          f"{J(POSTHOC, 'B/correlations/n')} uncorrected Spearman correlations, none of which reached |rho| >= "
          f"{J(POSTHOC, 'B/correlations/rho_crit')} (the two-sided {crit_pct} % level at n = {n_true}): "
          f"{J(POSTHOC, 'B/correlations/strongest/measure')} "
          f"({J(POSTHOC, 'B/measures/' + s['measure'] + '/description')}) of each training seed's shuffled graph "
          f"against {J(POSTHOC, 'B/correlations/strongest/deficit')}, the "
          f"validation {held30} deg/s deficit R2(`true`) - R2(`shuffled`); numbers, the training seeds. Validation "
          f"split, data seeds {v_lo}-{v_hi}, {J(POSTHOC, 'A/runs/true/t0/g30/n_trials')} rotation trials per held-out "
          f"speed and run; post-hoc and exploratory.", "",
          "All supporting figures (test split, validation and learning-rate selection, stimulus and visual input, "
          "video stills, alignment): [docs/FIGURES.md](docs/FIGURES.md).", ""]
    assert s["measure"] and s["deficit"]

    H(2, "How we got here (F0-F3)")
    f3 = _load(F3B3)["primary"]
    if f3["outcome"] != "replicated, not specific" or f3["bridge_off_selective"]:
        raise SystemExit("F3b-3 outcome differs from the README text")
    if rep["secondary"]["comparisons"]["true_vs_derange_r2"]["effect"]["mean"] <= 0:
        raise SystemExit("true did not beat derange in stage A; the README text differs")
    SM = "docs/results/SUMMARY_F0_F3.md"
    rd = "docs/results/"

    def lk(*names):
        return ", ".join(f"[{n}]({rd}{n}.md)" for n in names)
    pp = {t: J(F3B3, f"primary/perm/{t}/p_perm", "{:.3f}") for t in ("b3", "i1", "hg1", "hg4")}
    n_rand = JX(F3B3, ["primary/perm/b3/n_smaller", "primary/perm/b3/n_ge"], lambda a, b: a + b, "{}",
                "n_smaller + n_ge")
    L += [f"Synaptera v1 ([{T('CLAUDE.md', 'github.com/omeruk/synaptera', 'name')}](https://github.com/omeruk/synaptera)) flew a fly with a FlyWire brain and "
          "a FlyGym body but steered it with hand-made code; this repository hands steering to connectomes, step by "
          "step. Each phase has its own report; negative results carry equal weight "
          f"([summary of F0-F3]({rd}SUMMARY_F0_F3.md)).", "",
          "| phase | question | result | report |", "|---|---|---|---|",
          f"| F0 | does the ported v1 stack reproduce v1 (FlyWire brain and FlyGym body)? | yes; largest position "
          f"difference to v1 in the multi-seed acceptance {J(SUMMARY, 'f0_posdiff/values/0')} mm | "
          f"{lk('f0_acceptance', 'f0_multiseed')} |",
          f"| F1-F2 | can the BANC whole-brain graph replace the FlyWire brain? | no: with the unchanged reference "
          f"parameters the feeding motor neuron MN9 did not respond to the labellar sugar input of Shiu et al.'s "
          f"benchmark ({J(SUMMARY, 'f2_full/values/0')} Hz against {J(SUMMARY, 'f2_full/values/1')} Hz in FlyWire), "
          f"and uniform w_syn scaling ignited the network first; so the hybrid keeps the FlyWire brain | "
          f"{lk('f1_banc_smoke', 'f2_full_graph', 'f2_wsyn_calibration')} |",
          f"| {T(SM, 'F3 prep', 'name')} | which DN families are shared by name and pass the symmetry gate? | "
          f"{J(XWALK, 'summary/n_shared_gate_both')} of {J(XWALK, 'summary/shared_families')} shared families pass in "
          f"both datasets | {lk('f3_dn_crosswalk')} |",
          f"| F3a | does the BANC VNC (subgraph) turn DN activity into wing MN activity? | yes for wing power ({T(SM, 'DLM1-4', 'name')} "
          f"{J(SUMMARY, 'a_dlm/values/0')} Hz under bilateral DNg02 drive); ipsilateral steering by DNa04, DNa10 and "
          f"DNa15 in the VNC subgraph | {lk('f3a_banc_vnc', 'f3a3_vnc_subgraph')} |",
          f"| {T(SM, 'F3b-1', 'name')} | does a rotating grating reach the VNC through three DN families? | no: three "
          f"families were too few (highest applied DNa04 rate {J(SUMMARY, 'b1_max/values/1')} Hz against its lowest "
          f"steering rate of {J(SUMMARY, 'b1_max/values/2')} Hz) | {lk('f3b1_optomotor')} |",
          f"| {T(SM, 'F3b-2', 'name')} | does an ensemble of all shared, gate-passing families carry it? | "
          f"direction-selective, but so is the shuffled control | {lk('f3b2_dn_ensemble')} |",
          f"| {T(SM, 'F3b-3', 'name')} | is that specific to the true bridge mapping (fresh brain seeds, permutation "
          f"test against {n_rand} random derangements)? | the untrained spiking hybrid carried grating direction to "
          f"steering-MN asymmetry, replicated on fresh seeds and not direction-selective with the bridge off, but not "
          f"specific to the true mapping: permutation p {pp['b3']} (b3), {pp['i1']} (i1), {pp['hg1']} (hg1) and "
          f"{pp['hg4']} (hg4) | {lk('f3b3_specificity')} |",
          f"| F5 stage A | with training, does the real VNC wiring help? | yes on the test split (the primary test, "
          f"[Result in brief](#result-in-brief)) | {lk('f5_stageA_report')} |", "",
          "Side by side, and without a causal link: in stage A, with a different model (the trained rate VNC) and a "
          "different measure (test R2), the true bridge mapping beat `derange` (a secondary comparison) and the real "
          "VNC wiring beat `shuffled` (the primary test).", ""]

    H(2, "Key numbers")
    L += [f"Test split, mean over training seeds (standard deviations in "
          f"[docs/results/f5_stageA_report.md](docs/results/f5_stageA_report.md)). R2 at the training speeds is over "
          f"all trials at those speeds (rotation, static and grey); turn-sign accuracy is over the rotation trials at "
          f"the training speeds.", "",
          f"| arm | seeds | R2, training speeds | R2, {held90} deg/s | R2, {held30} deg/s | turn-sign accuracy, training speeds |",
          "|---|---:|---:|---:|---:|---:|"]
    for a in arms():
        r4 = [JX(REPORT, [f"summary/{a}/{k}/mean"], lambda v: round(v, 4) + 0.0, "{:.4f}",
                 "mean rounded to 4 decimals (+ 0.0: no negative zero)") for k in ("r2", "g90_r2", "g30_r2")]
        L.append(f"| `{a}` | {seeds_n(a)} | {r4[0]} | {r4[1]} | {r4[2]} | "
                 f"{J(REPORT, f'summary/{a}/sign_accuracy/mean', '{:.3f}')} |")
    L.append("")

    H(2, "How it works")
    if _get(TRANSFORMS, "transforms/affine/reflection") is not True:
        raise SystemExit("the affine has no reflection; the README text differs")
    win = _const("train/f5_stageA.py", "WINDOW")
    L += [img("docs/figures/fig0_pipeline.png",
              "Schematic in two rows of boxes joined by arrows: rotating drum, FlyGym eyes and FlyVis, FlyWire brain "
              "model (spiking, not trained, recorded once); then DN bridge (by cell type, fixed), BANC VNC subgraph "
              "(rate model, trained) and the fixed readout from steering motor neurons to the turn command dA, "
              "compared with the teacher dA*."), "",
          f"*Figure 0 | The stage-A pipeline: light blue is {T('scripts/readme_figures.py', 'FlyWire (fly 1)', 'name')}, "
          f"darker blue {T('scripts/readme_figures.py', 'BANC (fly 2)', 'name')}, grey neither; only the BANC VNC "
          f"subgraph is trained.*", "",
          "The brain and the nerve cord come from two different flies and are joined by DN cell type; see "
          "[Joining two flies: cell types and coordinates](#joining-two-flies-cell-types-and-coordinates).", "",
          f"- **Stimulus and brain.** A rotating grating (rot_left / rot_right at {held30}, {sp1}, {held90} and {sp2} "
          f"deg/s), a static grating and grey were rendered for the FlyGym eyes and passed through FlyVis to a visual "
          f"boundary layer that drives the {fw} brain, the unmodified spiking reference model of Shiu et al. "
          f"Each trial is {C('train/f5_stageA.py', 'T_TRIAL')} intervals of {C('train/f5_stageA.py', 'INTERVAL_MS', '{:g}')} "
          f"ms ({C('train/f5_stageA.py', 'N_GREY')} grey lead-in intervals, then the stimulus). The brain was run once "
          f"per trial and its DN spike counts recorded; stage A replays these records, so the brain is the same for "
          f"every arm and is never trained.",
          f"- **DN bridge.** The recorded FlyWire DN counts are turned into unit rates (window "
          f"{C('train/f5_stageA.py', 'LATENCIES', fn=lambda l: l['bridge_window_intervals'], formula='LATENCIES[bridge_window_intervals]')} "
          f"intervals, latency {C('train/f5_stageA.py', 'LATENCIES', fn=lambda l: l['bridge_latency_intervals'], formula='LATENCIES[bridge_latency_intervals]')} "
          f"interval) that clamp the BANC DNs of the same types (BANC DN state = unit rate / "
          f"{C('engine/rate_torch.py', 'R_IN', '{:g}')} Hz). DNg02 is clamped at a fixed baseline; every other BANC DN "
          f"is {zero_dn}.",
          f"- **VNC subgraph.** A rate model of the BANC VNC subgraph "
          f"({n_neu} neurons, {n_edge} edges), one update per interval. Trained: a gain per edge and a leak rate per neuron; the topology "
          f"and the synapse signs are fixed (prereg/14 section 2).",
          f"- **Readout (fixed).** dA = {C('train/f5_stageA.py', 'G1', '{:g}')} deg/Hz x "
          f"{C('engine/rate_torch.py', 'R_IN', '{:g}')} Hz x sum over types of sigma (s_left - s_right), one MN per "
          f"type and side, over the wing steering MN types {plus} (sigma +) and {minus} "
          f"(sigma -) (`LATERAL` in `scripts/f5_train.py`; the types of Lindsay et al., see the references).",
          f"- **Teacher and loss.** dA* = K_P v during the stimulus (v = +speed for rot_right, -speed for rot_left; {zero_teacher} "
          f"under static, grey and the lead-in), clipped at ±{C('train/f5_stageA.py', 'A_MAX', '{:g}')} deg, K_P = "
          f"{C('train/f5_stageA.py', 'A_MAX', '{:g}')} / {C('train/f5_stageA.py', 'V_MAX', '{:g}')} deg per deg/s. "
          f"Loss: the mean over every interval of ((dA - dA*) / {C('train/f5_stageA.py', 'A_MAX', '{:g}')} deg)^{sq}. "
          f"Metrics use the window mean over intervals {C('train/f5_stageA.py', 'WINDOW', fn=lambda w: w[0], formula='WINDOW[0]')}-"
          f"{C('train/f5_stageA.py', 'WINDOW', fn=lambda w: w[1] - 1, formula='WINDOW[1] - 1')} (the stimulus after its "
          f"first {C('train/f5_stageA.py', 'WINDOW', fn=lambda w: int((w[0] - _const('train/f5_stageA.py', 'N_GREY')) * _const('train/f5_stageA.py', 'INTERVAL_MS')), formula='(WINDOW[0] - N_GREY) x INTERVAL_MS')} ms).",
          "",
          "Arms (the same data, budget and test for each):", "",
          "- `true`: the VNC subgraph with its real wiring.",
          "- `shuffled`: the same subgraph rewired by degree-preserving edge swaps; the primary comparison arm.",
          "- `derange`: `true`, with the bridge feeding DN families into the wrong BANC targets.",
          "- `mlp`: an unconstrained two-layer ReLU network from the filtered bridge input to the readout MN states.",
          "- `linear`: one linear map from the same filtered input to the readout MN states.",
          f"- `bridge_off`: `true` with every bridged DN clamped to {zero_off}: the ablation and leakage check.",
          "- `untrained`: `true` at initialisation, evaluated without training.", "",
          "<details><summary>Full arm definitions (prereg/14 section 5, prereg/19 sections 2-3)</summary>", "",
          f"- `true`: the graphs of prereg/14 section 2.",
          f"- `shuffled`: degree-preserving directed double-edge swaps (a->b, c->d => a->d, c->b; no self-loops, no "
          f"duplicate pairs), {swaps} successful swaps; the "
          f"synapse count travels with the edge, the sign with the presynaptic neuron; swap RNG "
          f"`default_rng({C('train/f5_stageA.py', 'SHUFFLE_SEED0')} + t)` for training seed t.",
          "- `derange`: `true`, with the bridge family mapping replaced by the prereg/11 derangement `random_(t+1)`.",
          f"- `mlp`: two ReLU hidden layers of equal width h, then a linear map (no output nonlinearity) to the "
          f"{J(POSTHOC, 'B/targets/n')} readout MN states and the fixed readout; h is the smallest width with at least "
          f"the parameter count of `true`. Input: the {mlp_units} "
          f"units of the `true` bridge, each through exponential filters of "
          f"{', '.join(C('train/f5_stageA.py', 'FILTER_MS', '{:g}', fn=lambda f, i=i: f[i], formula=f'FILTER_MS[{i}]') for i in range(4))} ms.",
          "- `linear`: one linear map from the same filtered input to the readout MN states, then the fixed readout.",
          f"- `bridge_off`: `true` with every bridged DN clamped to {zero_off} (DNg02 kept).",
          "- `untrained`: `true` at initialisation, one deterministic evaluation on the test set.", "",
          f"Training seeds: {n_true} per trained arm, {seeds_n('bridge_off')} for `bridge_off`.", "", "</details>", ""]

    H(2, "What the fly sees")
    L += what_the_fly_sees(fw)

    H(2, "Joining two flies: cell types and coordinates")
    xw = _load(XWALK)
    dng02 = next(x for x in xw["families"] if x["family"] == "DNg02")
    if dng02["flywire"]["gate"] or not dng02["banc"]["gate"]:
        raise SystemExit("DNg02 does not fail the gate in FlyWire and pass in BANC; the README text differs")
    trf, wcj = _load(TRANSFORMS), _load(WARP)
    if not (trf["transforms"]["affine"]["reflection"] and trf["transforms"]["similarity"]["reflection"]):
        raise SystemExit("a transform has no reflection; the README text differs")
    if trf["transforms"]["similarity"]["outliers"] or trf["landmarks"]["dropped"]:
        raise SystemExit("the similarity rejected landmarks, or a landmark was dropped; the README text differs")
    if "mirror" not in wcj["note"]:
        raise SystemExit("the warp pipeline note does not name a mirror step")
    import csv as _csv
    with open(ROOT / "docs/figures/landmarks.csv", newline="") as f:
        _lm = list(_csv.DictReader(f))
    _sd = [_np_std([float(r[f"banc_{a}"]) for r in _lm]) for a in "xyz"]
    ax2 = sorted(sorted(range(3), key=lambda k: -_sd[k])[:2])
    axes_txt = " and ".join("xyz"[k] for k in ax2)
    XWMD, VP, P10, P11 = "docs/results/f3_dn_crosswalk.md", "scripts/viz_positions.py", "prereg/10_f3b2_dn_ensemble.md", \
        "prereg/11_f3b3_specificity.md"
    LAT = "train/f5_stageA.py"
    win_ms = C(LAT, "LATENCIES", fn=lambda l: int(l["bridge_window_intervals"] * _const(LAT, "INTERVAL_MS")),
               formula="LATENCIES[bridge_window_intervals] x INTERVAL_MS")
    lat_ms = C(LAT, "LATENCIES", fn=lambda l: int(l["bridge_latency_intervals"] * _const(LAT, "INTERVAL_MS")),
               formula="LATENCIES[bridge_latency_intervals] x INTERVAL_MS")
    tA, tS = "transforms/affine", "transforms/similarity"
    b_units = P(P10, "4", r"\*\*`ensemble`:\*\* (\d+) units")
    b_type = P(P10, "4", r"units \((\d+) by exact type")
    b_fam = P(P10, "4", r"by exact type, (\d+) by family\)")
    b_tgt = P(P10, "4", r"(\d+) BANC neurons \(\d+ left")
    b_tgt_l = P(P10, "4", r"BANC neurons \((\d+) left")
    b_tgt_r = P(P10, "4", r"BANC neurons \(\d+ left, (\d+) right")
    b_src = P(P10, "4", r"from (\d+) FlyWire neurons")
    sattolo_seed = P(P10, "5", r"Sattolo.s\s+algorithm, `numpy\.random\.default_rng\((\d+)\)`")
    rej = JX(TRANSFORMS, [f"{tA}/outliers"], lambda o: " and ".join(f"{x['cell_type']} ({x['side']})" for x in o), "{}",
             "outlier list: cell_type (side), joined with and")
    L += [f"The model joins the brain and the nerve cord by DN cell type; positions in space are used only for the "
          f"figures.", "",
          f"**By cell type (the model).** The brain ({fw}) and the brain and nerve cord of BANC ({banc}) come from two "
          f"different female flies (`docs/07_HYBRID_INTERFACE.md`), so DNs are matched by type, not by position. "
          f"FlyWire has {J(XWALK, 'summary/flywire_dn_neurons', '{:,}')} DNs in {J(XWALK, 'summary/flywire_families')} "
          f"type families, BANC {J(XWALK, 'summary/banc_dn_neurons', '{:,}')} in {J(XWALK, 'summary/banc_families')}; "
          f"{J(XWALK, 'summary/shared_families')} families are shared by name, and "
          f"{J(XWALK, 'summary/n_shared_gate_both')} of them pass the symmetry gate in both datasets. These form the DN "
          f"bridge (details below).", "",
          "<details><summary>Matching by cell type: rules and numbers</summary>", "",
          "- **Matching key:** exact `cell_type` first, then the family (`cell_type` without a single trailing "
          "`_<letter>`), else silent; left to left and right to right; centre neurons only to same-name centre "
          "neurons (`docs/07_HYBRID_INTERFACE.md`).",
          f"- **Shared families:** their neurons number {J(XWALK, 'summary/shared_neurons_flywire', '{:,}')} in FlyWire "
          f"and {J(XWALK, 'summary/shared_neurons_banc', '{:,}')} in BANC; "
          f"{JX(XWALK, ['summary/shared_count_equal'], len, '{}', 'len(summary/shared_count_equal)')} shared families "
          f"have identical left/right counts in both datasets.",
          f"- **Symmetry gate:** {T(XWMD, '|log2(R/L)| < 0.5', 'text')} of the input synapses, total and excitatory, on "
          f"each dataset's own graph; the bridge uses the shared families that pass it in both datasets.",
          f"- **The bridge:** {b_units} units ({b_type} by exact type, {b_fam} by family; "
          f"{J(RUNNER, 'design/bridge_unit_sides/left')} left, {J(RUNNER, 'design/bridge_unit_sides/right')} right); "
          f"{b_tgt} BANC targets ({b_tgt_l} left, {b_tgt_r} right) from {b_src} FlyWire neurons (prereg/10 section 4).",
          f"- **Rate transfer:** per BANC unit, the mean rate per neuron of the matched FlyWire DNs over a trailing "
          f"window of {win_ms} ms, applied {lat_ms} ms later; in the spiking hybrid (F3b) every BANC neuron of the unit "
          f"receives an independent Poisson process at that rate; in the stage-A rate model the BANC DN state is the "
          f"rate / {C('engine/rate_torch.py', 'R_IN', '{:g}')} Hz.",
          "- **Unmatched DNs:** unmatched BANC DNs are silent; unmatched FlyWire DNs are dropped.",
          f"- **Coverage gap:** {J(XWALK, 'summary/banc_mn_dn_families')} BANC DN families have direct synapses onto "
          f"wing, haltere, neck or TTMn motor neurons; {J(XWALK, 'summary/banc_mn_dn_families_absent_in_flywire')} of "
          f"them have no FlyWire family of the same name, and {J(XWALK, 'summary/banc_mn_dn_families_shared_gate_both')} "
          f"are shared and pass the gate in both datasets.",
          f"- **DNg02:** it fails the gate in FlyWire (log2(R/L) of the total input synapses "
          f"{JX(XWALK, ['families'], lambda fam: next(x for x in fam if x['family'] == 'DNg02')['flywire']['log2_total'], '{:+.2f}', 'flywire log2_total of family DNg02')}) "
          f"and passes in BANC, and it is silent in the FlyWire brain model; in the hybrid it is a fixed `HAND` "
          f"baseline ([summary of F0-F3](docs/results/SUMMARY_F0_F3.md)).",
          f"- **Controls on the mapping:** in F3b-2 a single random cycle over the shared, gate-passing families (a "
          f"derangement, drawn with Sattolo's algorithm, `default_rng({sattolo_seed})`, "
          f"side kept); in F3b-3 "
          f"{JX(F3B3, ['primary/perm/b3/n_smaller', 'primary/perm/b3/n_ge'], lambda a, b: a + b, '{}', 'n_smaller + n_ge')} "
          f"random derangements; in stage A the `derange` arm: the same rate streams and targets with the family "
          f"mapping deranged ({J(RUNNER, 'design/derange_units/t0')} units).",
          "- **Motor-neuron sides** were audited against their exit nerves "
          "([Methods and safeguards](#methods-and-safeguards)).", "", "</details>", "",
          f"**In space (the figures).** For the figures only, FlyWire positions were mapped into BANC space. Our own "
          f"transform, an affine fitted to DN-type landmarks, is a visualisation scaffold for viewing only "
          f"(`scripts/viz_positions.py`) and includes a reflection. The anatomy figure, the activity figure and the "
          f"video do not use it: they draw the BANC team's warped FlyWire skeletons "
          f"({T(WARP, 'FAFB14 -> JRC2018F -> mirror -> BANC, elastix + thin-plate spline', 'text')}) and BANC "
          f"skeletons. Our affine places the brain-model positions of Figure S15 and is compared with the BANC team's "
          f"warp in Figure 6. None of this enters the model.", "",
          img("docs/figures/fig6_alignment.png",
              "Four panels: landmark points of the two datasets in BANC coordinates joined by residual lines; strips of "
              "residual norms for the affine and the similarity; a cumulative distribution of the distance between our "
              "affine and the BANC team's warp; and a scatter of that distance against the distance from the landmark "
              "centroid."), "",
          f"**Figure 6 | Our FlyWire -> BANC affine, used only for the figures, against its landmarks and against the "
          f"BANC team's warp.** a, The landmarks in BANC coordinates on the two axes with the largest spread of the "
          f"BANC landmarks ({axes_txt}): BANC landmark points (blue), FlyWire landmark points after our affine (sky "
          f"blue) and a grey line per landmark for its residual; circles, control; squares, check; crosses, the "
          f"rejected outliers. This is a 2-D view of 3-D residuals. b, Residual norms (µm) of the affine and the "
          f"similarity for the control points (crosses, the affine's rejected outliers) and the check points; bars, "
          f"medians. c, Distance between our affine-placed FlyWire soma and the root node of the BANC team's warped "
          f"skeleton, as a cumulative distribution over the DNs used; dashed, the median. d, The same distance against "
          f"the distance of the warped root from the centroid of the fitted landmarks. "
          f"{J(TRANSFORMS, 'landmarks/n')} landmarks ({J(TRANSFORMS, 'landmarks/n_control')} control, "
          f"{J(TRANSFORMS, 'landmarks/n_check')} check); c and d, {J(WARP, 'used')} DNs.", "",
          "<details><summary>Positions, landmarks, the two transforms and the check against the warp</summary>", "",
          f"- **Positions:** FlyWire brain-model neurons at their soma ({T(VP, 'voxels x (4, 4, 40) nm', 'text')}) "
          f"where present ({J(TRANSFORMS, 'brain/soma', '{:,}')} of {J(TRANSFORMS, 'brain/n', '{:,}')}), else at the "
          f"centroid of their synapses ({J(TRANSFORMS, 'brain/centroid_synapses', '{:,}')}) or a fallback centroid "
          f"({J(TRANSFORMS, 'brain/centroid_fallback', '{:,}')}); BANC VNC neurons at their root position "
          f"({J(TRANSFORMS, 'vnc/root_position', '{:,}')}), else at the synapse-weighted mean position of their "
          f"partners ({J(TRANSFORMS, 'vnc/filled')}), else dropped ({J(TRANSFORMS, 'vnc/dropped')}).",
          f"- **Landmarks:** one per bridged DN type and side matched by exact type (family matches are not used): the "
          f"FlyWire mean soma against the BANC mean root position of the type's neurons with a nucleus id, as type "
          f"means {T(VP, '(neurons within a type have no one-to-one match)', 'text')}; "
          f"{J(TRANSFORMS, 'landmarks/n_types')} types, none dropped.",
          f"- **Hold-out:** {JX(TRANSFORMS, ['landmarks/holdout_frac'], lambda v: round(100 * v), '{}', 'fraction x 100')} % "
          f"of the landmark types, both sides together (`default_rng({J(TRANSFORMS, 'landmarks/split_seed')})`), are "
          f"check points: {J(TRANSFORMS, 'landmarks/n_check_types')} types.",
          f"- **Fit:** least squares on the control points, refitted after rejecting every point whose residual exceeds "
          f"{J(TRANSFORMS, 'sigma_k', '{:g}')} x the RMSE of the kept points, until none is rejected; the affine "
          f"rejected {rej}, the similarity none.", "",
          "| | affine | similarity |", "|---|---:|---:|",
          f"| parameters | {J(TRANSFORMS, f'{tA}/n_params')} | {J(TRANSFORMS, f'{tS}/n_params')} |",
          f"| RMSE, control points (µm) | {J(TRANSFORMS, f'{tA}/rmse_control_um', '{:.1f}')} | "
          f"{J(TRANSFORMS, f'{tS}/rmse_control_um', '{:.1f}')} |",
          f"| RMSE, check points (µm) | {J(TRANSFORMS, f'{tA}/rmse_check_um', '{:.1f}')} | "
          f"{J(TRANSFORMS, f'{tS}/rmse_check_um', '{:.1f}')} |",
          f"| outliers removed | {J(TRANSFORMS, f'{tA}/outliers_removed')} | {J(TRANSFORMS, f'{tS}/outliers_removed')} |",
          f"| determinant | {J(TRANSFORMS, f'{tA}/determinant', '{:.3f}')} | {J(TRANSFORMS, f'{tS}/determinant', '{:.3f}')} |",
          f"| scale | {J(TRANSFORMS, f'{tA}/scale', '{:.3f}')} | {J(TRANSFORMS, f'{tS}/scale', '{:.3f}')} |",
          f"| side check, left DNs | {J(TRANSFORMS, f'{tA}/side_check/left/fraction_same_side', '{:.3f}')} | "
          f"{J(TRANSFORMS, f'{tS}/side_check/left/fraction_same_side', '{:.3f}')} |",
          f"| side check, right DNs | {J(TRANSFORMS, f'{tA}/side_check/right/fraction_same_side', '{:.3f}')} | "
          f"{J(TRANSFORMS, f'{tS}/side_check/right/fraction_same_side', '{:.3f}')} |", "",
          f"Side check: the fraction of the FlyWire left ({J(TRANSFORMS, f'{tA}/side_check/left/n')}) and right "
          f"({J(TRANSFORMS, f'{tA}/side_check/right/n')}) DNs that land on the same BANC side, by the majority label "
          f"of the {J(TRANSFORMS, 'side_k')} nearest BANC neurons with a nucleus id.",
          "- **Reflection:** both determinants are negative, so both maps include a mirror image (this follows from "
          "the definition of the determinant); this is consistent with the mirror step of the BANC team's pipeline "
          "named in `docs/figures/warp_check.json`.",
          f"- **Unit check of the native skeletons:** read in {J(WARP, 'native_unit')} (not voxels), the native "
          f"skeleton's root lies within {J(WARP, 'root_tol_um', '{:g}')} µm of the FlyWire soma for "
          f"{J(WARP, 'root_within_tol')} DNs (median root-soma distance {J(WARP, 'root_soma_um_median', '{:.2f}')} µm).",
          f"- **Excluded:** {J(WARP, 'excluded_other_tree/n')} DNs ({J(WARP, 'excluded_other_tree/cell_types')} types): "
          f"their warped file is another skeleton than the native one, so its root is not the soma.",
          f"- **Our affine against the warp,** over the DNs used: median {J(WARP, 'error/median_um', '{:.1f}')} µm, RMSE "
          f"{J(WARP, 'error/rmse_um', '{:.1f}')} µm, 95th percentile {J(WARP, 'error/p95_um', '{:.1f}')} µm, maximum "
          f"{J(WARP, 'error/max_um', '{:.1f}')} µm; median by landmark set: control "
          f"{J(WARP, 'error_by_landmark_set/control/median_um', '{:.1f}')} µm (n = "
          f"{J(WARP, 'error_by_landmark_set/control/n')}), rejected outliers "
          f"{J(WARP, 'error_by_landmark_set/control_outlier/median_um', '{:.1f}')} µm (n = "
          f"{J(WARP, 'error_by_landmark_set/control_outlier/n')}), check "
          f"{J(WARP, 'error_by_landmark_set/check/median_um', '{:.1f}')} µm (n = "
          f"{J(WARP, 'error_by_landmark_set/check/n')}), no landmark "
          f"{J(WARP, 'error_by_landmark_set/none/median_um', '{:.1f}')} µm (n = "
          f"{J(WARP, 'error_by_landmark_set/none/n')}); mean signed difference x "
          f"{J(WARP, 'mean_signed_diff_um/x', '{:+.2f}')}, y {J(WARP, 'mean_signed_diff_um/y', '{:+.2f}')}, z "
          f"{J(WARP, 'mean_signed_diff_um/z', '{:+.2f}')} µm.",
          f"- **Distance dependence, as measured:** Spearman rho "
          f"{J(WARP, 'distance_dependence/spearman_rho', '{:.2f}')} (p = "
          f"{J(WARP, 'distance_dependence/spearman_p', '{:.1e}')}) between the error and the distance from the "
          f"landmark centroid; slope "
          f"{JX(WARP, ['distance_dependence/slope_um_per_100um'], lambda v: v / 100, '{:.3f}', 'slope_um_per_100um / 100')} "
          f"µm of error per µm of distance; "
          f"median error by distance third, nearest to farthest: "
          f"{J(WARP, 'distance_dependence/tertiles/0/median_err_um', '{:.1f}')}, "
          f"{J(WARP, 'distance_dependence/tertiles/1/median_err_um', '{:.1f}')} and "
          f"{J(WARP, 'distance_dependence/tertiles/2/median_err_um', '{:.1f}')} µm.",
          "- **Verified:** `scripts/verify_report.py` (`check_alignment`) refits both transforms from the committed "
          "landmarks on every run and reproduces the stored matrices, the rejected outliers and the hold-out. The side "
          "check is the one value it cannot recompute: it needs the BANC neuron table and the position of every "
          "FlyWire DN, which are not committed.",
          "- **Which positions each figure uses:** Figure 2 (`scripts/viz_anatomy.py`, `load_circuit_lines`), "
          "Figure 3 (`scripts/viz_activity_figure.py`, `circuit_lines`) and the video and GIF "
          "(`scripts/viz_video.py`, through `circuit_lines`) draw the BANC team's warped FlyWire skeletons "
          "(`fafb_banc`) and BANC skeletons (`banc`); our affine places the brain-model positions of Figure S15 "
          "(`scripts/viz_preview.py`) and is compared with the warp in Figure 6 (`check` step of "
          "`scripts/viz_anatomy.py`).", "", "</details>", ""]
    for code_file, needle in (("scripts/viz_anatomy.py", 'man["kind"].isin(["fafb_banc", "banc"])'),
                              ("scripts/viz_activity_figure.py", 'man["kind"].isin(["fafb_banc", "banc"])'),
                              ("scripts/viz_video.py", "AF.circuit_lines("),
                              ("scripts/viz_preview.py", '(brain["xyz_nm"] @ M.T + t)')):
        if needle not in (ROOT / code_file).read_text():
            raise SystemExit(f"{code_file} no longer contains {needle!r}; the position sources in the README differ")

    H(2, "Training")
    rn, rc = _load(RUNNER)["design"], _load(RECORDS)
    if rn["n_trainable_graph"] != rn["n_edges"] + rn["n_neurons"] - rn["n_inputs"]:
        raise SystemExit("trainable parameters are not one gain per edge and one leak rate per non-input neuron")
    if rn["mlp_params"] < rn["n_trainable_graph"] or not rc["gate5"]["bit_identical"]:
        raise SystemExit("mlp smaller than true, or gate 5 not bit-identical; the README text differs")
    if rc["completion"]["input_checks_passed"] != rc["completion"]["input_checks_total"]:
        raise SystemExit("not every input check passed; the README text differs")
    if holm["not_run"] != ["B", "C"]:
        raise SystemExit("stages B and C are not both 'not run'")
    vs, cu = _load(VALID), _load(CURVES)
    first_eval = cu["eval_every"] * cu["batch"]
    if vs["summary"]["true"]["trials_to_090"] != {"mean": first_eval, "sd": 0.0}:
        raise SystemExit("true did not reach the set accuracy at the first evaluation in every seed")
    bo = vs["summary"]["bridge_off"]
    if bo["n_censored_090"] != bo["n_seeds"]:
        raise SystemExit("bridge_off reached the set accuracy in some seed; the README text differs")
    if min(vs["summary"], key=lambda a: vs["summary"][a]["best_loss"]["mean"]) != "mlp":
        raise SystemExit("mlp does not have the lowest validation loss")
    if list(vs["guard"]["pause_reasons"]) != ["GPU temperature"]:
        raise SystemExit("a guard pause had another reason")
    sec = rep["secondary"]["comparisons"]["true_vs_shuffled_samples"]
    if sec["test"]["mean_d"] <= 0 or "> 0 favours true" not in sec["label"]:
        raise SystemExit("the samples-to-set-accuracy comparison does not favour true")
    import numpy as _np
    lin_r2 = _np.array([r["evaluations"]["r2"] for r in cu["runs"] if r["arm"] == "linear" and r["selected_rate"]]).mean(0)
    fig_src = (ROOT / "scripts/readme_figures.py").read_text()
    r2_lo = float(re.search(r"FIG5_R2_LIM = \((-?[\d.]+),", fig_src).group(1))
    if not (lin_r2 < r2_lo).any():
        raise SystemExit("the linear mean stays inside the Figure 5 c axis; the caption says it leaves it")
    P14S = "prereg/14_f5_training.md"
    cur_rows = [("A", ("rate BANC VNC subgraph", "recorded FlyWire DN spike counts (spiking reference brain R) through the "
                       "bridge", "open-loop dA"), "run"),
                ("B", ("rate FlyWire brain + rate VNC (VNC warm-started from A, same arm and training seed)",
                       "frozen FlyVis 1.2.0 -> boundary layer", "open-loop dA"), "not run"),
                ("C", ("as B, warm-started from B", "online frozen FlyVis in the docs/08 closed loop",
                       "behaviour cloning, then 5 DAgger rounds (beta = 0)"), "not run"),
                ("D", ("optional", "-", "needs its own pre-registration before any D run; never part of the section 8 "
                       "verdict"), "optional, not run")]
    table = ["| stage | trained part | input | task | status |", "|---|---|---|---|---|"]
    for st, cells, status in cur_rows:
        line = f"| {st} | " + " | ".join(cells) + " |"
        T(P14S, line, "text")                                         # the row of prereg/14 section 3, verbatim
        table.append(f"| {st} | " + " | ".join(T(P14S, c, "text") if any(ch.isdigit() for ch in c) else c
                                               for c in cells) + f" | {status} |")
    seeds_tr = J(VALID, "summary/true/n_seeds")
    L += [f"Stage A trains only the nerve cord. The brain is fixed: its DN spike counts were recorded once and are "
          f"replayed through the bridge. The rate model of the BANC VNC subgraph learns a gain per edge and a leak rate "
          f"per neuron, each within fixed bounds (prereg/14 section 2), so the wiring itself, topology and synapse "
          f"signs, never changes: the comparison asks whether the real wiring is a better substrate for learning the "
          f"turn command than the same neurons with shuffled wiring. Every arm gets the same data, budget, "
          f"learning-rate selection rule and test (H1, prereg/14 section 1).", "",
          "The curriculum (prereg/14 section 3):", ""] + table + [""] + training_data(
        sp1, sp2, held30, held90, v_lo, v_hi, t_lo, t_hi) + [
          "What was trained:", "",
          f"- **The graph:** {J(RUNNER, 'design/n_neurons', '{:,}')} neurons, {J(RUNNER, 'design/n_edges', '{:,}')} "
          f"edges and {J(RUNNER, 'design/n_trainable_graph', '{:,}')} trainable parameters, one gain per edge and one "
          f"leak rate per neuron; the {J(RUNNER, 'design/n_inputs', '{:,}')} clamped DN inputs have no leak rate.",
          f"- **Fair controls:** `mlp` has {J(RUNNER, 'design/mlp_params', '{:,}')} parameters (width "
          f"{J(RUNNER, 'design/mlp_width', '{:,}')}, the smallest with at least the parameter count of `true`, prereg/14 "
          f"section 5); `linear` has {J(RUNNER, 'design/linear_params', '{:,}')}; `shuffled` keeps every neuron's in- "
          f"and out-degree, and every edge keeps its synapse count and presynaptic sign.",
          f"- **The same input for every arm:** the brain was recorded once, "
          f"{J(RECORDS, 'completion/n_plan', '{:,}')} records, with {J(RECORDS, 'completion/input_checks_passed')} of "
          f"{J(RECORDS, 'completion/input_checks_total')} input checks passed, and the gate-5 check reproduced a "
          f"reference record bit for bit (prereg/14 gate 5).", "",
          "<details><summary>How it was trained</summary>", "",
          "- **Data:** the data seeds, records and trials of each split are in the training-data table above.",
          f"- **Optimiser and budget:** Adam, batch {J(RUNNER, 'design/batch')}, {J(RUNNER, 'design/steps', '{:,}')} "
          f"steps, so {JX(RUNNER, ['design/steps', 'design/batch'], lambda a, b: a * b, '{:,}', 'steps x batch')} "
          f"training trials presented per run.",
          f"- **Evaluation and checkpoint:** the full validation set every {J(RUNNER, 'design/eval_every')} steps; the "
          f"checkpoint with the lowest validation loss is kept (prereg/14 section 6).",
          f"- **Learning rate:** the grid {{{', '.join(J(RUNNER, f'design/lr_grid/{i}', '{:g}') for i in range(3))}}}; "
          f"per arm, the rate with the lowest final validation loss at training seed {seed0} is used for every seed.",
          f"- **Training seeds:** {seeds_tr} per trained arm, {J(VALID, 'summary/bridge_off/n_seeds')} for `bridge_off`.",
          f"- **Memory:** a sparse edge gradient (sddmm, prereg/18), so the full graph trains on an "
          f"{J(BENCH, 'machine/gpu_total_gib', '{:.0f}')} GiB laptop GPU; the smoke run peaked at "
          f"{J(RUNNER, 'smoke/gpu_peak_allocated_gib', '{:.1f}')} GiB of GPU memory.",
          f"- **One GPU job at a time,** through a guard (`scripts/f5_train_guard.py`) with GPU memory capped at a "
          f"fraction {J(BENCH, 'policy/gpu_fraction_cap')}, a RAM cap of {J(BENCH, 'policy/ram_cap')} and a pause at a "
          f"GPU temperature of {J(BENCH, 'policy/kill_temp_c')} °C. This followed the parallel benchmark, which ended in "
          f"a driver fault ({J(BENCH, 'incident/nvlddmkm_events', '{:,}')} nvlddmkm events and an unexpected shutdown; "
          f"`docs/results/f5_benchmark.json`, incident).",
          f"- **Guard record:** {J(VALID, 'guard/job_ends_rc0')} of {J(VALID, 'guard/job_ends')} jobs ended normally; "
          f"{J(VALID, 'guard/pauses')} pauses, all for GPU temperature; from {J(VALID, 'guard/first')} to "
          f"{J(VALID, 'guard/last')}. After a pause a run resumes from its latest checkpoint (written every "
          f"{J(RUNNER, 'design/ckpt_every')} steps).",
          f"- **Replicate (prereg/19):** `true` at seed {seed0} and its selected rate, trained a second time, agreed with "
          f"the first run at every evaluation to "
          f"{J(VALID, 'replicate/max_abs_diff_over_evaluations/loss', '{:.1e}')} in validation loss and "
          f"{J(VALID, 'replicate/max_abs_diff_over_evaluations/r2', '{:.1e}')} in R2, with equal turn-sign accuracy.",
          "- **Test:** once per run, on the CPU, from the selected checkpoint (see "
          "[Methods and safeguards](#methods-and-safeguards)).", "", "</details>", "",
          "What happened (validation split, descriptive):", "",
          "- **Selected learning rates:** " + ", ".join(f"`{a}` {J(VALID, f'selected_lr/{a}', '{:g}')}"
                                                        for a in vs["selected_lr"]) + ".", "",
          img("docs/figures/fig5_training.png",
              "Four panels: final validation loss against learning rate per arm with the selected rate circled; "
              "validation loss and validation R2 over training with mean lines and bands per arm; and dots for the "
              "training trials each seed needed to reach the set turn-sign accuracy."),
          "",
          f"**Figure 5 | Stage-A training on the validation split: learning-rate selection, learning curves and the "
          f"trials to the set turn-sign accuracy.** a, Learning-rate selection at training seed {seed0}: final "
          f"validation loss per arm and learning rate (log scale); circles mark the selected rate. b, Validation loss "
          f"over training trials presented at the selected rate (log scale); lines, means over training seeds; bands, "
          f"± 1 SD (cut at the axis where they fall below it). c, Validation R2 of the window mean, same layout; "
          f"dotted, `untrained` (one evaluation at initialisation); the `linear` mean leaves the axis at some "
          f"evaluations (its R2 swings widely; prereg/20 limitation c). d, Training trials until the validation "
          f"turn-sign accuracy first reached {J(CURVES, 'set_accuracy', '{:.2f}')}, one marker per training seed "
          f"(vertical jitter only); ticks at the evaluation interval "
          f"({JX(CURVES, ['eval_every', 'batch'], lambda a, b: a * b, '{:,}', 'eval_every x batch')} trials); open "
          f"markers at the right edge, later than the axis end or never (`bridge_off` never). Validation split, data seeds "
          f"{v_lo}-{v_hi} ({J(RUNNER, 'design/n_val_trials')} trials per evaluation); {seeds_tr} training seeds per "
          f"arm, {J(VALID, 'summary/bridge_off/n_seeds')} for `bridge_off`.", "",
          f"- **Training trials to the set turn-sign accuracy** (mean over seeds): " + ", ".join(
              f"`{a}` {J(VALID, f'summary/{a}/trials_to_090/mean', '{:,.0f}')}"
              for a in ("true", "shuffled", "derange", "mlp", "linear")) +
          f"; `bridge_off` never ({J(VALID, 'summary/bridge_off/n_censored_090')} of "
          f"{J(VALID, 'summary/bridge_off/n_seeds')} seeds). `true` reached it at the first evaluation in every seed. "
          f"The evaluation interval limits the resolution, and turn-sign accuracy is near ceiling (prereg/20 "
          f"limitation b).",
          f"- **Registered secondary comparison** of samples to set accuracy, `true` against `shuffled` (prereg/14 "
          f"section 8.3: counted on the validation evaluations during training, compared in the test report; no "
          f"Holm, never a verdict): in favour of `true`, one-sided p = "
          f"{J(REPORT, 'secondary/comparisons/true_vs_shuffled_samples/test/p_greater', '{:.4f}')} "
          f"({J(REPORT, 'secondary/comparisons/true_vs_shuffled_samples/test/count_greater')}/"
          f"{J(REPORT, 'secondary/comparisons/true_vs_shuffled_samples/test/n_patterns')}).",
          f"- `mlp` reached the lowest validation loss (selected checkpoints, mean "
          f"{J(VALID, 'summary/mlp/best_loss/mean', '{:.4f}')}).",
          "- The `shuffled` validation loss was still falling at the end of the budget (prereg/20 limitation a; see "
          "[Limitations](#limitations)).", ""] + validation_tables(vs, v_lo, v_hi, seed0)

    H(2, "Methods and safeguards")
    rl = _load(RATE_LIF)
    if not (rl["agreement"] and rl["c1"]["agrees"] and rl["c2"]["agrees"]):
        raise SystemExit("the rate-LIF comparison is not 'qualitative agreement'; the README text differs")
    if "qualitative agreement" not in (ROOT / "docs/results/f5_rate_lif.md").read_text():
        raise SystemExit("f5_rate_lif.md does not state qualitative agreement")
    relabelled = _load(SIDE)["section1"]["relabelled_readout_types"]
    stage_a_types = set(_const("scripts/f5_train.py", "LATERAL"))
    if stage_a_types & set(relabelled):
        raise SystemExit(f"stage-A readout types in the relabelled list: {sorted(stage_a_types & set(relabelled))}")
    post = _load(SIDE_POST)
    if post["label"] != "post-audit" or set(post["swapped_types"]) | set(post["not_recomputable_types"]) != set(relabelled):
        raise SystemExit("the post-audit swapped / not-recomputable types do not cover the relabelled list")
    joined = lambda v: ", ".join(v)
    relab_n = JX(SIDE, ["section1/relabelled_readout_types"], len, "{}", "len(section1/relabelled_readout_types)")
    relab = JX(SIDE, ["section1/relabelled_readout_types"], joined, "{}", "comma-separated list")
    swapped = JX(SIDE_POST, ["swapped_types"], joined, "{}", "comma-separated list")
    not_rec = JX(SIDE_POST, ["not_recomputable_types"], joined, "{}", "comma-separated list")
    lk = pr["guards"]["leakage"]
    titles = []
    for p in sorted((ROOT / "prereg").glob("*.md")):
        if p.name[:2].isdigit() and 14 <= int(p.name[:2]) <= 21:
            t = p.read_text().splitlines()[0].lstrip("# ").strip()
            titles.append(f"- [`{p.name}`](prereg/{p.name}): {T(f'prereg/{p.name}', t, 'title')}")
    L += ["- **Pre-registration.** The hypotheses, model, arms, budget, metrics, primary test and verdict rules were "
          "committed before any stage-A code or run (prereg/14); the addenda prereg/15-20 were committed before the "
          "runs or the test evaluation they concern, and prereg/21 fixed the post-hoc analyses before they were "
          "computed.",
          f"- **Sealed test split.** The test data seeds ({t_lo}-{t_hi}) are listed in `configs/f5_splits.json`; only "
          f"`final-eval` may read them. Each run was evaluated once, on the CPU "
          f"({C('train/f5_stageA.py', 'TEST_THREADS')} threads, deterministic) from its selected checkpoint; a "
          f"`test.lock` file is created before the evaluation, so a second one is refused.",
          f"- **Primary test.** One-sided exact paired sign-flip test of mean d over all "
          f"{J(REPORT, 'primary/test/n_patterns')} sign patterns, Holm-adjusted over stages A, B and C.",
          f"- **Guards.** Ceiling: the verdict is qualified if both `true` and `shuffled` have mean test R2 >= "
          f"{J(REPORT, 'primary/guards/ceiling/threshold')} (not triggered). Leakage: the verdict is withheld if "
          f"`bridge_off` has a mean test turn-sign accuracy outside the binomial {bin_pct} % interval around chance "
          f"({J(REPORT, 'primary/guards/leakage/interval/lo')}-{J(REPORT, 'primary/guards/leakage/interval/hi')} for "
          f"{J(REPORT, 'primary/guards/leakage/interval/n')} rotation trials); it was "
          f"{J(REPORT, 'primary/guards/leakage/mean_sign_accuracy')} (not triggered).",
          "- **Controls.** The arms above: a shuffled graph, a deranged bridge, an ablation, two unconstrained "
          "networks and the untrained model.",
          "- **Reproducibility.** Training ran on the GPU, where runs agree only to float32 rounding (prereg/19); the "
          "test numbers come from the CPU evaluation and are exactly reproducible from the saved checkpoints.",
          "- **Every number recomputed.** `scripts/verify_report.py` recomputes the numbers of the result reports "
          "from the records and checks them against the text.",
          f"- **Rate model against the spiking VNC.** Before training, the rate model was checked against the spiking "
          f"LIF VNC on the {T('docs/results/SUMMARY_F0_F3.md', 'F3a-3', 'name')} protocol (prereg/15): qualitative agreement, with power recruitment agreeing in "
          f"{J(RATE_LIF, 'c1/n_agree')} of {J(RATE_LIF, 'c1/n')} items and the steering sign in "
          f"{J(RATE_LIF, 'c2/n_agree')} of {J(RATE_LIF, 'c2/n')}.",
          f"- **Motor-neuron sides audited.** The side of every motor neuron was audited against its exit nerve "
          f"(prereg/13): {relab_n} readout types of the earlier F3 analyses were relabelled ({relab}); the post-audit "
          f"recomputation swapped {swapped} and could not recompute {not_rec}. None of the {n_types} stage-A readout "
          f"types is in the relabelled list.",
          "- **Errata, not edits.** Factual errors in pre-registration texts are corrected in an errata file, never by "
          "editing the pre-registration ([prereg/errata_01_post_hoc.md](prereg/errata_01_post_hoc.md)).", "",
          "<details><summary>Pre-registrations of stage A</summary>", ""] + titles + ["", "</details>", ""]

    H(2, "Limitations")
    lim = {x["id"]: x for x in rep["limitations"]}
    assert list(lim) == list("abcde")
    L += ["Recorded in prereg/20 section 1 before the test was unsealed:", "",
          "- a. Budget and learning rate: `true` and `shuffled` both selected the largest learning rate of the grid, "
          "and the `shuffled` validation loss was still falling at the end of the budget, so d may partly reflect "
          "learning speed.",
          "- b. Turn-sign accuracy is at ceiling, so it separates the arms little.",
          "- c. `linear` was unstable at every learning rate; its values are noisy.",
          "- d. The `mlp` learning-rate choice was a near tie.",
          "- e. `true` was close to the ceiling guard on the validation split.", "",
          "Further:", "",
          f"- Stages B and C (trained brain, then closed loop) were not run, so no closed-loop behaviour was tested; "
          f"the prereg/20 exploratory follow-up (longer training, a larger learning rate) was not run either.",
          "- The post-hoc results (Figure 4) are exploratory.",
          f"- The `untrained` arm already had a test turn-sign accuracy of "
          f"{J(REPORT, 'summary/untrained/sign_accuracy/mean', '{:.3f}')}, so turn-sign accuracy does not separate the "
          f"arms; R2, which also scores the size of the turn, is the primary metric.",
          f"- The readout uses only the {n_types} "
          f"MN types named above, one neuron per type and side.",
          "- The model was not compared with neural or behavioural recordings from flies.",
          "- The brain and the nerve cord come from two different flies; the bridge assumes that DNs of the same type "
          "carry the same signal in both, which was not tested, and only DN types shared by name and passing the "
          "symmetry gate are bridged.",
          "- Wing power comes from a fixed, hand-set DNg02 baseline, because DNg02 is silent in the FlyWire brain "
          "model; steering is the only part driven by the brain.",
          "- Everything here is a simulation: open loop, one steering task (optomotor yaw), stage A only.", ""]
    assert not rep["exploratory"]["run"] and holm["not_run"] == ["B", "C"]

    H(2, "Reproduce")
    d_rec, d_tr, d_gd = doc("scripts/f5_stageA_records.py"), doc("scripts/f5_train.py"), doc("scripts/f5_train_guard.py")
    d_val, d_rep, d_ver = doc("scripts/f5_stageA_validation.py"), doc("scripts/f5_stageA_report.py"), doc("scripts/verify_report.py")
    d_fig = doc("scripts/readme_figures.py")
    gate_parts = " ".join(_const("scripts/f5_train.py", "GATE_PARTS")) + " flyvis_render summary"
    _rec(gate_parts, dict(code="scripts/f5_train.py:GATE_PARTS", note="plus the flyvis_render and summary parts of "
                          "the gates parser"), "command")
    cpu, gpu = J(BENCH, "machine/cpu"), J(BENCH, "machine/gpu")
    L += ["<details><summary>Environment, data and commands</summary>", "",
          "- **Environment.** `bash env/create_neurofly.sh` builds the conda environment `neurofly` (x86_64 Linux) "
          "from `env/neurofly_x86_64.yml`. Run every command with `env -u PYTHONPATH` in that environment.",
          "- **External data.** The brain model, the FlyWire and BANC tables and the BANC edge lists are listed with "
          "their origin and SHA-256 in `configs/data_sources.yaml`; they are never committed (`banc/`, "
          "`brain_model/`, `data/`). The download script `scripts/fetch_data.py` named in `CLAUDE.md` has not been "
          "carried over to this repository yet (`MIGRATION_CHECKLIST.md`, item 31), so the files have to be fetched "
          "from the origins listed there. Steps 1-5 need them, and so do most checks of `scripts/verify_report.py`, which "
          "recompute from the records in `runs/` (not committed); step 6 reads only committed files.",
          f"- **Hardware used** (as recorded in `docs/results/f5_benchmark.json`): CPU {cpu}; GPU {gpu}; WSL2.", "",
          "1. Stage-A brain records (CPU):", "", "```",
          d_rec("env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python scripts/f5_stageA_records.py run --render-workers N --brain-workers M"),
          d_rec("env -u PYTHONPATH python scripts/f5_stageA_records.py manifest"), "```", "",
          f"2. Setup and gates (parts: {gate_parts}):", "", "```",
          d_tr("env -u PYTHONPATH python scripts/f5_train.py setup"),
          d_tr("env -u PYTHONPATH python scripts/f5_train.py gates --part records"), "```", "",
          f"3. Training on the GPU, through the guard (one GPU job at a time): the learning-rate grid at training seed "
          f"{seed0} for each arm, the rate selection, then the other seeds and the replicate:", "", "```",
          d_gd("env -u PYTHONPATH python scripts/f5_train_guard.py --log runs/f5/A/guard.log -- \\"),
          "    " + d_gd("python scripts/f5_train.py run --arm true --seed 0 --lr 1e-3"),
          d_tr("env -u PYTHONPATH python scripts/f5_train.py select-lr --arm true"),
          d_tr("env -u PYTHONPATH python scripts/f5_train.py run --arm true --seed 3"),
          d_tr("env -u PYTHONPATH python scripts/f5_train.py run --arm true --seed 0 --replicate"), "```", "",
          "4. The test evaluation, once per run, on the CPU (the only reader of the test seeds):", "", "```",
          d_tr("env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python scripts/f5_train.py final-eval --arm true --seed 3"), "```",
          "",
          "5. Reports and the post-hoc analyses:", "", "```",
          d_val("env -u PYTHONPATH python scripts/f5_stageA_validation.py"),
          d_rep("env -u PYTHONPATH python scripts/f5_stageA_report.py"),
          "env -u PYTHONPATH CUDA_VISIBLE_DEVICES= python scripts/f5_stageA_posthoc.py predict",
          "env -u PYTHONPATH python scripts/f5_stageA_posthoc.py report",
          "env -u PYTHONPATH python scripts/f5_stageA_posthoc.py figure", "```", "",
          "6. Figure 1 and this README (they read only committed JSON files):", "", "```",
          d_fig("env -u PYTHONPATH python scripts/readme_figures.py"),
          "env -u PYTHONPATH python scripts/readme_text.py", "```", "",
          "7. Checks:", "", "```",
          d_ver("env -u PYTHONPATH python scripts/verify_report.py"),
          "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 env -u PYTHONPATH python -m pytest tests/", "```", "",
          "</details>", ""]
    ph_src = (ROOT / "scripts/f5_stageA_posthoc.py").read_text()
    for sub in ("predict", "report", "figure"):
        assert f'"{sub}"' in ph_src
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 env -u PYTHONPATH python -m pytest tests/" in (ROOT / "CLAUDE.md").read_text()

    H(2, "Repository layout")
    layout = [("body/", "body/__init__.py"), ("connectome/", "connectome/__init__.py"),
              ("coupling/", "coupling/__init__.py"), ("engine/", "engine/__init__.py"), ("motor/", "motor/__init__.py"),
              ("sense/", "sense/__init__.py"), ("tasks/", "tasks/__init__.py")]
    rows = {d: T(f, ast.get_docstring(ast.parse((ROOT / f).read_text())).split("\n")[0], "text") for d, f in layout}
    L += ["<details><summary>Top-level folders</summary>", "",
          f"- `body/`: {rows['body/']}",
          "- `configs/`: configuration files, the data sources (`data_sources.yaml`), the parameter registry "
          "(`assumptions.json`) and the data splits (`f5_splits.json`).",
          f"- `connectome/`: {rows['connectome/']}",
          f"- `coupling/`: {rows['coupling/']}",
          "- `docs/`: design documents; `docs/results/` (result reports), `docs/figures/` (figures and their record), "
          "`docs/provenance/` (the Synaptera v1 provenance record).",
          f"- `engine/`: {rows['engine/']}",
          "- `env/`: the conda environment and the script that builds it.",
          f"- `motor/`: {rows['motor/']}",
          "- `prereg/`: the pre-registrations, never edited after commit.",
          "- `scripts/`: run, report, figure and verification scripts.",
          f"- `sense/`: {rows['sense/']}",
          f"- `tasks/`: {rows['tasks/']}",
          "- `tests/`: the test suite.",
          "- `train/`: the stage-A training code, the gates, graph shuffles and checkpoints.",
          "- Not committed: `banc/`, `brain_model/`, `data/` (external data) and `runs/` (run outputs).", "",
          "</details>", ""]

    H(2, "Data, credits and licences")
    N = "docs/provenance/synaptera_v1_NOTICE.md"
    TP = "docs/provenance/synaptera_v1_THIRD_PARTY.md"
    xref = dict(source="Crossref (api.crossref.org/works/<doi>), checked 2026-10-27; no copy in the repository")

    def linked(doi):
        return f"doi:[{doi}](https://doi.org/{doi})"
    refs = [
        ("FlyWire connectome", T(TP, "Dorkenwald, S. et al. (2024). *Nature* 634, 124–138. " + linked("10.1038/s41586-024-07558-y"))),
        ("FlyWire annotations", T(TP, "Schlegel, P. et al. (2024). *Nature* 634, 139–152. " + linked("10.1038/s41586-024-07686-5"))),
        ("FAFB electron-microscopy volume", X("Zheng, Z. et al. (2018). *Cell* 174, 730–743.e22. " + linked("10.1016/j.cell.2018.06.019"), xref)),
        ("FlyWire transmitter predictions (synapse signs)", T(TP, "Eckstein, N. et al. (2024). *Cell* 187, 2574–2594. " + linked("10.1016/j.cell.2024.03.016"))),
        ("BANC connectome", X("Bates, A. S. et al. (2026). *Nature* 656, 957–970. " + linked("10.1038/s41586-026-10735-w"), xref)
         + "; data: Harvard Dataverse " + linked(T("scripts/viz_anatomy.py", "10.7910/DVN/7WTH1N", "reference"))),
        ("brain model", T(TP, "Shiu, P. K. et al. (2024). *Nature* 634, 210–219. " + linked("10.1038/s41586-024-07763-9"))),
        ("FlyGym / NeuroMechFly v2 (body, eyes)", T(TP, "Wang-Chen, S. et al. (2024). *Nature Methods* 21, 2353–2362. " + linked("10.1038/s41592-024-02497-y"))),
        ("FlyVis (visual network)", T(TP, "Lappalainen, J. K. et al. (2024). *Nature* 634, 1132–1140. " + linked("10.1038/s41586-024-07939-3"))),
        ("Brian2 (spiking simulation)", T(TP, "Stimberg, M., Brette, R. & Goodman, D. F. M. (2019). *eLife* 8, e47314. " + linked("10.7554/eLife.47314"))),
        ("wing steering motor system (readout MN types)", X("Lindsay, T., Sustar, A. & Dickinson, M. (2017). *Current Biology* 27, 345–358. " + linked("10.1016/j.cub.2016.12.018"), xref)),
    ]
    L += ["- **Code:** MIT ([LICENSE](LICENSE)).",
          f"- **Data:** {fw} data are {cc_nc}; BANC {banc} data are {cc_by} and need their attribution. "
          "Neither is redistributed here.",
          f"- **Figures:** {cc_nc}, because they contain FlyWire-derived content; per-figure credits, sources and "
          "hashes in [docs/figures/README.md](docs/figures/README.md).",
          "- **Third-party software and its licences:** "
          "[docs/provenance/synaptera_v1_NOTICE.md](docs/provenance/synaptera_v1_NOTICE.md).", "",
          "Works this repository uses:", ""] + [f"- {what}: {ref}" for what, ref in refs] + [""]

    H(2, "Authorship")
    L += [AUTHOR, "", CREDIT, "", "Contact: https://github.com/omeruk", ""]

    H(2, "How to cite")
    L += ["Cite this repository with the metadata in [CITATION.cff](CITATION.cff) (GitHub shows it as \"Cite this "
          "repository\"), and cite the data and models it uses (above).", ""]

    for t in toc:
        assert t in headings, t
    return "\n".join(L)


# ── gallery ───────────────────────────────────────────────────────────────────────────────────────────────────────
def readme_figures(readme_text):
    """The figure numbers of the "Figure N |" captions of the README, as "lo-hi" when they have no gap."""
    nums = sorted({int(n) for n in re.findall(r"(?m)^\*{1,2}Figure (\d+) \|", readme_text)})
    if not nums:
        raise SystemExit("the README has no figure captions")
    if nums == list(range(nums[0], nums[-1] + 1)):
        return f"{nums[0]}-{nums[-1]}" if len(nums) > 1 else f"{nums[0]}"
    return ", ".join(map(str, nums[:-1])) + f" and {nums[-1]}"


def gallery(readme_text):
    """docs/FIGURES.md; the range of README figures is read from the "Figure N |" captions of readme_text."""
    CUR["file"] = "docs/FIGURES.md"
    in_readme = readme_figures(readme_text)
    sup = ROOT / "docs" / "figures" / "supplementary"
    groups = [("Test split", "test_"), ("Validation and learning-rate selection", "validation_"),
              ("Stimulus and visual input", "stimulus_"), ("Video stills", "video_"), ("Alignment", "alignment_")]
    files = sorted(p.name for p in sup.glob("*.png"))
    assert sorted(f for _, g in groups for f in files if f.startswith(g)) == files, "a file in no group"
    rep, val = _load(REPORT), _load(VALID)
    held30, held90 = C("train/f5_stageA.py", "HELD_OUT", fn=lambda h: h["secondary"], formula="HELD_OUT['secondary']"), \
        C("train/f5_stageA.py", "HELD_OUT", fn=lambda h: h["primary"], formula="HELD_OUT['primary']")
    sp1 = C("train/f5_stageA.py", "TRAIN_CONDITIONS", fn=lambda c: sorted({v for _, v in c if v})[0],
            formula="lowest speed of TRAIN_CONDITIONS")
    sp2 = C("train/f5_stageA.py", "TRAIN_CONDITIONS", fn=lambda c: sorted({v for _, v in c if v})[1],
            formula="highest speed of TRAIN_CONDITIONS")
    v_lo, v_hi = val_range()
    if not all(x["final_loss"] < 0 for x in val["true_minus_shuffled"]["per_seed"].values()):
        raise SystemExit("not every seed has a lower final validation loss for true; the caption says every seed")
    sel = rep["selected_lr"]
    stills = _const("scripts/viz_video.py", "STILLS")
    grid = ", ".join(C("train/f5_stageA.py", "LR_GRID", "{:g}", fn=lambda g, i=i: g[i], formula=f"LR_GRID[{i}]")
                     for i in range(3))
    tf = "transforms/affine"
    seed0 = P("prereg/14_f5_training.md", "6", r"final validation loss\s+at training seed (\d+) is used")
    pv = "scripts/f5_stageA_preview.py"
    stim_seed = P(pv, "docstring", r"training data seed (\d+) \(phase")
    stim_phase = P(pv, "docstring", r"\(phase (\d+); never")
    st = [C("scripts/viz_video.py", "STILLS", fn=lambda v, i=i: v[i][1], formula=f"STILLS[{i}][1]") for i in range(3)]
    act = [C("scripts/viz_video.py", "STILLS", fn=lambda v, i=i: v[i][0], formula=f"STILLS[{i}][0]") for i in range(3)]
    ci_pct = JX(REPORT, ["primary/effect/level"], lambda v: round(100 * v), "{}", "level x 100")
    cap = {
        "test_a_true_vs_shuffled_paired_r2.png": (
            f"On the test split, `true` had a higher R2 than `shuffled` at the training speeds in every training seed.",
            f"Test R2 at the training speeds ({sp1} and {sp2} deg/s) per training seed, `shuffled` (orange) and `true` "
            f"(blue); grey lines join the two runs of a seed. The title gives the pre-registered verdict, the "
            f"one-sided exact p, the Holm-adjusted p and mean d with its {ci_pct} % bootstrap interval. Test split, "
            f"{J(REPORT, 'summary/true/r2/n')} training seeds per arm."),
        "test_b_test_r2_per_arm.png": (
            "Test R2 at the training speeds for every arm.",
            "Dots, training seeds; bars, arm means (printed above each bar); n, the number of training seeds of the arm. "
            "Test split."),
        "test_c_generalisation_true_vs_shuffled.png": (
            f"Across drum speeds `true` and `shuffled` had the same turn-sign accuracy on the test split, and differed "
            f"in R2 mainly at {held30} deg/s.",
            f"Left, test turn-sign accuracy; right, test R2; at {held30} deg/s (held out), {sp1} and {sp2} deg/s "
            f"(training) and {held90} deg/s (held out); dots, training seeds; lines join the arm means; `shuffled` "
            f"orange, `true` blue. Test split, {J(REPORT, 'summary/shuffled/r2/n')} training seeds per arm."),
        "validation_a_lr_grid.png": (
            f"Learning-rate grid at training seed {seed0}: validation loss and R2 over training, per arm and rate.",
            f"Rows, the arms `true`, `shuffled`, `derange`, `mlp`, `linear` and `bridge_off`; left, validation loss "
            f"(log scale); right, validation R2 of the window mean; x, training trials presented; colours, the "
            f"learning rates {grid}, the selected one marked in the legend. Validation split (used for selection); "
            f"training seed {seed0}. Figure 5 in the README shows the same validation evaluations."),
        "validation_b_selected_lr_all_seeds.png": (
            "Validation R2 over training at each arm's selected learning rate.",
            f"One panel per arm, its rate in the title; thin grey lines, single training seeds; red, their mean. "
            f"Validation split (used for selection); {J(REPORT, 'summary/derange/r2/n')} "
            f"training seeds per arm, {J(REPORT, 'summary/bridge_off/r2/n')} for `bridge_off`. Figure 5 in the README shows the same validation evaluations."),
        "validation_c_arm_comparison.png": (
            "The arms at their selected learning rates over training.",
            f"Left, validation loss (log scale); centre, turn-sign accuracy (red dashed line, the "
            f"{C('train/f5_stageA.py', 'SET_ACCURACY', '{:.2f}')} set-accuracy level); right, R2 of the window mean; "
            f"x, training trials presented; lines, means over training seeds; shading, ± s.d.; dotted, `untrained` at "
            f"initialisation; colours as in the legend (they differ from Figures 1 and 4). Validation split (used for "
            f"selection). Figure 5 in the README shows the same validation evaluations."),
        "validation_d_true_vs_shuffled_final_val_loss.png": (
            "The final validation loss was lower for `true` than for `shuffled` in every training seed (descriptive).",
            f"Final validation loss (step {C('train/f5_stageA.py', 'STEPS', '{:,}')}) per training seed, `true` against "
            f"`shuffled`; lines join the two runs of a seed. Validation split (used for selection); descriptive, not "
            f"the pre-registered test; {J(REPORT, 'summary/mlp/r2/n')} training seeds per arm."),
        "stimulus_eye_images_rot_left_v060_seed0.png": (
            f"What the two eyes see: raw eye images during leftward rotation at {sp1} deg/s.",
            "Luminance seen by the left (top row) and right (bottom row) FlyGym eye in the grey lead-in and at three "
            f"times after the drum starts (times in the panel titles). Training data seed {stim_seed}, grating phase {stim_phase}. "
            "Figure 7 in the [README](../README.md#what-the-fly-sees) redraws this render: two of its frames (the grey "
            "lead-in and the first stimulus time) in panel a, with a space-time plot of the middle image row."),
        "stimulus_t4t5_rot_left_v060_seed0.png": (
            f"FlyVis T4/T5 boundary rates during leftward rotation at {sp1} deg/s.",
            "Heat maps of the rates (Hz, colour bar) of the T4 and T5 boundary neurons over time, left and right eye, "
            f"rows grouped by subtype (T4a-d, T5a-d); the cyan line marks the drum start. Training data seed {stim_seed}. "
            "Figure 7 in the [README](../README.md#what-the-fly-sees) redraws this render in panel b, on one colour "
            "scale shared with Figure S11 and against time from drum onset."),
        "stimulus_t4t5_rot_left_v120_seed0.png": (
            f"FlyVis T4/T5 boundary rates during leftward rotation at {sp2} deg/s.",
            f"As Figure S9. Training data seed {stim_seed}. Figure 7 in the [README](../README.md#what-the-fly-sees) "
            f"does not redraw this heat map; the per-subtype means of this render enter its panel c only."),
        "stimulus_t4t5_rot_right_v060_seed0.png": (
            f"FlyVis T4/T5 boundary rates during rightward rotation at {sp1} deg/s.",
            f"As Figure S9. Training data seed {stim_seed}. Figure 7 in the [README](../README.md#what-the-fly-sees) "
            f"redraws this render in panel b, on one colour scale shared with Figure S9 and against time from drum "
            f"onset."),
        "video_still_act1_+0300ms.png": (
            f"Act {act[0]} of the circuit video at +{st[0]} ms: change from the pre-stimulus baseline during rightward "
            f"rotation.",
            f"Each drawn neuron is coloured by its change from its own baseline (colour bars: FlyWire DNs in Hz, BANC "
            f"neurons in model units); the inset trace is the turn command dA; dashed, the teacher dA*. "
            f"{C('scripts/viz_video.py', 'SPEED')} deg/s, mean over the validation data seeds {v_lo}-{v_hi}."),
        "video_still_act2_+0300ms.png": (
            f"Act {act[1]} of the circuit video at +{st[1]} ms: direction preference, rightward minus leftward rotation.",
            f"Orange prefers rightward, blue leftward (colour bars as in Figure S12); the inset shows dA for both "
            f"directions. {C('scripts/viz_video.py', 'SPEED')} deg/s, mean over the validation data seeds "
            f"{v_lo}-{v_hi}."),
        "video_still_act2_+0800ms.png": (
            f"Act {act[2]} of the circuit video at +{st[2]} ms.",
            "As Figure S13, later in the trial."),
        "alignment_positions_affine_by_source.png": (
            "FlyWire brain-model neurons mapped into BANC space by our affine transform (a visualisation scaffold).",
            f"Front (x-z), side (y-z) and top (x-y) projections in BANC coordinates (µm); brain neurons coloured by the "
            f"source of their position (soma, synapse centroid, fallback centroid), the BANC VNC subgraph in light grey, "
            f"landmarks as crosses (BANC mean soma) and circles (transformed FlyWire mean soma). The title gives the "
            f"landmark fit of the affine in `docs/figures/transforms.json`: RMSE "
            f"{J(TRANSFORMS, f'{tf}/rmse_control_um', '{:.1f}')} µm on the control points and "
            f"{J(TRANSFORMS, f'{tf}/rmse_check_um', '{:.1f}')} µm on the check points, determinant "
            f"{J(TRANSFORMS, f'{tf}/determinant', '{:.3f}')}. Against the BANC team's warp (`docs/figures/warp_check.json`: "
            f"our affine FlyWire soma against the root node of the BANC team's warped skeleton) the median distance was "
            f"{J(WARP, 'error/median_um', '{:.1f}')} µm over {J(WARP, 'error/n')} DNs."),
    }
    assert sel and set(cap) == set(files)
    assert [s[1] for s in stills] == [int(x.split("_+")[1][:4]) for x in sorted(f for f in files if f.startswith("video_"))]
    L = ["# Supporting figures", "",
         "Every file of `docs/figures/supplementary/`, by group. Sources, scripts and hashes: "
         f"[figures/README.md](figures/README.md). Figures {in_readme} are in the [README](../README.md).", ""]
    k = 0
    for title, g in groups:
        L += [f"## {title}", ""]
        for f in [x for x in files if x.startswith(g)]:
            k += 1
            head, body = cap[f]
            if g == "validation_":
                assert "Validation split (used for selection)" in body
            L += [img(f"figures/supplementary/{f}", f"Figure S{k}: {head}"), "",
                  f"**Figure S{k} | {head}** {body}", ""]
    return "\n".join(L)


def build() -> dict:
    """The three outputs as text, keyed by their path; nothing is written."""
    ENTRIES.clear()
    rd = readme()
    texts = {"README.md": rd, "docs/FIGURES.md": gallery(rd)}
    for t in texts.values():
        for bad in ("@", "omerf", "C:\\", "/mnt/c"):
            assert bad not in t, bad
    merged = {}
    for e in ENTRIES:
        key = (e["string"], json.dumps(e["source"], sort_keys=True))
        m = merged.setdefault(key, dict(string=e["string"], files=[], kind=e["kind"], source=e["source"]))
        if e["file"] not in m["files"]:
            m["files"].append(e["file"])
    nums = sorted(merged.values(), key=lambda m: (m["kind"], m["string"]))
    for m in nums:
        for f in m["files"]:
            assert m["string"] in texts[f], (m["string"], f)
    texts["docs/readme_numbers.json"] = json.dumps(dict(
        note="every number, name with a version, reference and command printed in README.md and docs/FIGURES.md, with "
             "its source; generated by scripts/readme_text.py", entries=nums), indent=1, ensure_ascii=False) + "\n"
    return texts


def check(texts) -> int:
    """0 if every committed output equals the generator output, else 1 with the first differing line."""
    for rel, new in texts.items():
        path = ROOT / rel
        old = path.read_text() if path.exists() else None
        if old == new:
            continue
        if old is None:
            print(f"{rel}: missing")
            return 1
        a, b = old.splitlines(), new.splitlines()
        k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        print(f"{rel}: line {k + 1} differs\n  committed: {a[k] if k < len(a) else '<end of file>'}\n"
              f"  generated: {b[k] if k < len(b) else '<end of file>'}")
        return 1
    print("README.md, docs/FIGURES.md and docs/readme_numbers.json equal the generator output")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check", action="store_true", help="compare with the committed files; write nothing")
    a = ap.parse_args(argv)
    if a.check:
        return check(build())
    for p in (README, GALLERY, NUMBERS):
        if p.exists() and not a.overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite to replace it")
    texts = build()
    for rel, t in texts.items():
        (ROOT / rel).write_text(t)
    n = len(json.loads(texts["docs/readme_numbers.json"])["entries"])
    print(f"{n} entries; README {len(texts['README.md']):,} chars, gallery {len(texts['docs/FIGURES.md']):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
