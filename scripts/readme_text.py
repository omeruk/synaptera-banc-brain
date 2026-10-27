"""The repository README, the figure gallery and the list of every number they print.

    env -u PYTHONPATH python scripts/readme_text.py [--overwrite]

Writes README.md, docs/FIGURES.md and docs/readme_numbers.json. No number is typed here (CLAUDE.md rule 11): each one
is read from a committed JSON (docs/results/*.json, docs/figures/*.json, configs/f5_splits.json), from a code constant
(parsed from the source, not imported), or from the text of a pre-registration, and docs/readme_numbers.json records
for each printed string the file(s) it appears in and its source. References and the commands of the Reproduce
section are copied from repository files and checked against them. Refuses to overwrite an existing output without
--overwrite.
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README, GALLERY, NUMBERS = ROOT / "README.md", ROOT / "docs" / "FIGURES.md", ROOT / "docs" / "readme_numbers.json"
REPORT = "docs/results/f5_stageA_report.json"
POSTHOC = "docs/results/f5_stageA_posthoc.json"
VALID = "docs/results/f5_stageA_validation.json"
BENCH = "docs/results/f5_benchmark.json"
SPLITS = "configs/f5_splits.json"
TRANSFORMS = "docs/figures/transforms.json"
WARP = "docs/figures/warp_check.json"
P19 = "prereg/19_f5_gpu_reproducibility_bridge_units_readout.md"
REPO_URL = "https://github.com/omeruk/synaptera-banc-brain"
AUTHOR = "Ömer Faruk Topal, Department of Geomatics Engineering, Istanbul Technical University."
CREDIT = "Designed and directed by Ömer Faruk Topal; implemented with the help of Claude Code (Anthropic)."

ENTRIES = []
CUR = {"file": None}
_JSON = {}


def _load(f):
    if f not in _JSON:
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


def img(src, alt):
    return f'<img src="{src}" width="100%" alt="{alt.replace("`", "")}">'


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
    toc = ["Result in brief", "The stage-A circuit", "Activity on the circuit", "Post-hoc analyses", "Key numbers",
           "How it works", "Methods and safeguards", "Limitations", "Reproduce", "Repository layout",
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
          f"1 - SSE / SST of the window-mean turn command dA against the teacher dA*; d is the per-seed difference "
          f"R2(`true`) - R2(`shuffled`). Mean d was {J(REPORT, 'primary/effect/mean', '{:.4f}')} (95 % bootstrap "
          f"interval {J(REPORT, 'primary/effect/lo', '{:.4f}')} to {J(REPORT, 'primary/effect/hi', '{:.4f}')}), "
          f"one-sided exact sign-flip p = {J(REPORT, 'primary/test/p_greater', '{:.5f}')} "
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
              "with mlp highest; every arm per condition, where true is highest at 30 deg/s."), "",
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
              "Four panels: gain ratio against drum speed per arm; validation R2 at 30 deg/s, true above shuffled for "
              "every seed; z-scores of 18 graph measures of the true graph against 10 shuffled graphs; a scatter of "
              "one graph measure against the per-seed deficit."), "",
          f"**Figure 4 | Post-hoc, exploratory: on the validation data seeds `true` kept its advantage at {held30} "
          f"deg/s, and its graph lay outside the range of the shuffles on {n_out} of {n_meas} structure measures.** "
          f"a, Speed tuning: gain ratio G(v) = mean sign(dA*) dA / (K_P v) of the window-mean output against drum "
          f"speed, per arm; lines, means over the training seeds; small dots, single seeds; `untrained`, one run; "
          f"dashed line, the teacher (G = 1); the held-out speeds are shaded. b, Validation R2 at {held30} deg/s, "
          f"`true` against `shuffled` per training seed (lines pair the seeds), with the primary post-hoc test in the "
          f"panel. c, Structure of the `true` VNC graph against the shuffled graphs: per measure S1-S18 "
          f"([definitions](docs/results/f5_stageA_posthoc.md)), z = (value - shuffled mean) / shuffled SD; grey, the "
          f"shuffled graphs; blue, `true` (filled where it lies outside the shuffled range); symmetric-log axis. d, The "
          f"strongest structure-deficit correlation, selected as the largest |rho| of "
          f"{J(POSTHOC, 'B/correlations/n')} uncorrected Spearman correlations, none of which reached |rho| >= "
          f"{J(POSTHOC, 'B/correlations/rho_crit')} (the two-sided 5 % level at n = {n_true}; "
          f"{J(POSTHOC, 'B/correlations/n_at_or_above_crit')} reached it): {J(POSTHOC, 'B/correlations/strongest/measure')} "
          f"({J(POSTHOC, 'B/measures/' + s['measure'] + '/description')}) of each training seed's shuffled graph "
          f"against {J(POSTHOC, 'B/correlations/strongest/deficit')}, the "
          f"validation {held30} deg/s deficit R2(`true`) - R2(`shuffled`); numbers, the training seeds. Validation "
          f"split, data seeds {v_lo}-{v_hi}, {J(POSTHOC, 'A/runs/true/t0/g30/n_trials')} rotation trials per held-out "
          f"speed and run; post-hoc and exploratory.", "",
          "All supporting figures (test split, validation and learning-rate selection, stimulus and visual input, "
          "video stills, alignment): [docs/FIGURES.md](docs/FIGURES.md).", ""]
    assert s["measure"] and s["deficit"]

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
    win = _const("train/f5_stageA.py", "WINDOW")
    L += ["```mermaid", "flowchart LR",
          f'  S["Drum stimulus: rotating grating, rot_left / rot_right, static, grey"] --> E["FlyGym eyes, FlyVis, '
          f'visual boundary layer"]',
          f'  E --> Bn["{fw} brain: spiking reference model (Shiu et al.), not trained; per-interval DN spike counts '
          f'recorded once per trial"]',
          f'  Bn --> D["DN bridge: {units} units into {targets} BANC DNs, replayed from the records"]',
          f'  D --> V["BANC VNC subgraph, rate model: edge gains and leak rates trained, topology fixed"]',
          f'  V --> M["Fixed readout: {J(POSTHOC, "B/targets/n")} steering MNs to the turn command dA"]',
          f'  M --> L["Loss against the teacher dA*"]',
          "```", "",
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
          f"is 0.",
          f"- **VNC subgraph.** A rate model of the BANC VNC subgraph "
          f"({n_neu} neurons, {n_edge} edges), one update per interval. Trained: a gain per edge and a leak rate per neuron; the topology "
          f"and the synapse signs are fixed (prereg/14 section 2).",
          f"- **Readout (fixed).** dA = {C('train/f5_stageA.py', 'G1', '{:g}')} deg/Hz x "
          f"{C('engine/rate_torch.py', 'R_IN', '{:g}')} Hz x sum over types of sigma (s_left - s_right), one MN per "
          f"type and side, over the wing steering MN types {plus} (sigma +) and {minus} "
          f"(sigma -) (`LATERAL` in `scripts/f5_train.py`; types of Lindsay et al. 2017).",
          f"- **Teacher and loss.** dA* = K_P v during the stimulus (v = +speed for rot_right, -speed for rot_left; 0 "
          f"under static, grey and the lead-in), clipped at ±{C('train/f5_stageA.py', 'A_MAX', '{:g}')} deg, K_P = "
          f"{C('train/f5_stageA.py', 'A_MAX', '{:g}')} / {C('train/f5_stageA.py', 'V_MAX', '{:g}')} deg per deg/s. "
          f"Loss: the mean over every interval of ((dA - dA*) / {C('train/f5_stageA.py', 'A_MAX', '{:g}')} deg)^2. "
          f"Metrics use the window mean over intervals {C('train/f5_stageA.py', 'WINDOW', fn=lambda w: w[0], formula='WINDOW[0]')}-"
          f"{C('train/f5_stageA.py', 'WINDOW', fn=lambda w: w[1] - 1, formula='WINDOW[1] - 1')} (the stimulus after its "
          f"first {C('train/f5_stageA.py', 'WINDOW', fn=lambda w: int((w[0] - _const('train/f5_stageA.py', 'N_GREY')) * _const('train/f5_stageA.py', 'INTERVAL_MS')), formula='(WINDOW[0] - N_GREY) x INTERVAL_MS')} ms).",
          f"- **Training.** Adam, batch {C('train/f5_stageA.py', 'BATCH')}, "
          f"{C('train/f5_stageA.py', 'STEPS', '{:,}')} steps, the learning rate chosen per arm at training seed {seed0} from "
          f"{{{', '.join(C('train/f5_stageA.py', 'LR_GRID', '{:g}', fn=lambda g, i=i: g[i], formula=f'LR_GRID[{i}]') for i in range(3))}}} "
          f"by the lowest final validation loss, and the checkpoint with the lowest validation loss kept. Data seeds: "
          f"training {JX(SPLITS, ['train'], min, '{}', 'min of train')}-{JX(SPLITS, ['train'], max, '{}', 'max of train')}, "
          f"validation {v_lo}-{v_hi}, test {t_lo}-{t_hi}.", "",
          "Arms (the same data, budget and test for each):", "",
          "- `true`: the VNC subgraph with its real wiring.",
          "- `shuffled`: the same subgraph rewired by degree-preserving edge swaps; the primary comparison arm.",
          "- `derange`: `true`, with the bridge feeding DN families into the wrong BANC targets.",
          "- `mlp`: an unconstrained two-layer ReLU network from the filtered bridge input to the readout MN states.",
          "- `linear`: one linear map from the same filtered input to the readout MN states.",
          "- `bridge_off`: `true` with every bridged DN clamped to 0: the ablation and leakage check.",
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
          "- `bridge_off`: `true` with every bridged DN clamped to 0 (DNg02 kept).",
          "- `untrained`: `true` at initialisation, one deterministic evaluation on the test set.", "",
          f"Training seeds: {n_true} per trained arm, {seeds_n('bridge_off')} for `bridge_off`.", "", "</details>", ""]

    H(2, "Methods and safeguards")
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
          f"`bridge_off` has a mean test turn-sign accuracy outside the binomial 95 % interval around chance "
          f"({J(REPORT, 'primary/guards/leakage/interval/lo')}-{J(REPORT, 'primary/guards/leakage/interval/hi')} for "
          f"{J(REPORT, 'primary/guards/leakage/interval/n')} rotation trials); it was "
          f"{J(REPORT, 'primary/guards/leakage/mean_sign_accuracy')} (not triggered).",
          "- **Controls.** The arms above: a shuffled graph, a deranged bridge, an ablation, two unconstrained "
          "networks and the untrained model.",
          "- **Reproducibility.** Training ran on the GPU, where runs agree only to float32 rounding (prereg/19); the "
          "test numbers come from the CPU evaluation and are exactly reproducible from the saved checkpoints.",
          "- **Every number recomputed.** `scripts/verify_report.py` recomputes the numbers of the result reports "
          "from the records and checks them against the text.", "",
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
    rows = {d: ast.get_docstring(ast.parse((ROOT / f).read_text())).split("\n")[0] for d, f in layout}
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
    xref = dict(source="Crossref (api.crossref.org/works/<doi>), checked 2026-10-27")
    refs = [
        ("FlyWire connectome", T(N, "Dorkenwald, S. et al. (2024). *Nature* 634, 124–138. doi:10.1038/s41586-024-07558-y")),
        ("FlyWire annotations", T(N, "Schlegel, P. et al. (2024). *Nature* 634, 139–152. doi:10.1038/s41586-024-07686-5")),
        ("FAFB electron-microscopy volume", X("Zheng, Z. et al. (2018). *Cell* 174, 730–743.e22. doi:10.1016/j.cell.2018.06.019", xref)),
        ("FlyWire transmitter predictions (synapse signs)", T(TP, "Eckstein, N. et al. (2024). *Cell* 187, 2574–2594. doi:[10.1016/j.cell.2024.03.016](https://doi.org/10.1016/j.cell.2024.03.016)")),
        ("BANC connectome", X("Bates, Phelps, Kim, Yang et al. (2026). *Nature* 656, 957–970. doi:10.1038/s41586-026-10735-w", xref)
         + "; data: Harvard Dataverse " + T("docs/figures/README.md", "doi:10.7910/DVN/7WTH1N", "reference")),
        ("brain model", T(N, "Shiu, P. K. et al. (2024). *Nature* 634, 210–219. doi:10.1038/s41586-024-07763-9")),
        ("FlyGym / NeuroMechFly v2 (body, eyes)", T(N, "Wang-Chen, S. et al. (2024). *Nature Methods*. doi:10.1038/s41592-024-02497-y")),
        ("FlyVis (visual network)", T(N, "Lappalainen, J. K. et al. (2024). *Nature* 634, 1132–1140. doi:10.1038/s41586-024-07939-3")),
        ("Brian2 (spiking simulation)", T(N, "Stimberg, M., Brette, R. & Goodman, D. F. M. (2019). *eLife* 8, e47314. doi:10.7554/eLife.47314")),
        ("wing steering motor system (readout MN types)", X("Lindsay, T., Sustar, A. & Dickinson, M. (2017). *Current Biology* 27, 345–358. doi:10.1016/j.cub.2016.12.018", xref)),
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
def gallery():
    CUR["file"] = "docs/FIGURES.md"
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
    cap = {
        "test_a_true_vs_shuffled_paired_r2.png": (
            f"On the test split, `true` had a higher R2 than `shuffled` at the training speeds in every training seed.",
            f"Test R2 at the training speeds ({sp1} and {sp2} deg/s) per training seed, `shuffled` (orange) and `true` "
            f"(blue); grey lines join the two runs of a seed. The title gives the pre-registered verdict, the "
            f"one-sided exact p, the Holm-adjusted p and mean d with its 95 % bootstrap interval. Test split, "
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
            f"training seed {seed0}."),
        "validation_b_selected_lr_all_seeds.png": (
            "Validation R2 over training at each arm's selected learning rate.",
            f"One panel per arm, its rate in the title; thin grey lines, single training seeds; red, their mean. "
            f"Validation split (used for selection); {J(REPORT, 'summary/derange/r2/n')} "
            f"training seeds per arm, {J(REPORT, 'summary/bridge_off/r2/n')} for `bridge_off`."),
        "validation_c_arm_comparison.png": (
            "The arms at their selected learning rates over training.",
            f"Left, validation loss (log scale); centre, turn-sign accuracy (red dashed line, the "
            f"{C('train/f5_stageA.py', 'SET_ACCURACY', '{:.2f}')} set-accuracy level); right, R2 of the window mean; "
            f"x, training trials presented; lines, means over training seeds; shading, ± s.d.; dotted, `untrained` at "
            f"initialisation; colours as in the legend (they differ from Figures 1 and 4). Validation split (used for "
            f"selection)."),
        "validation_d_true_vs_shuffled_final_val_loss.png": (
            "The final validation loss was lower for `true` than for `shuffled` in every training seed (descriptive).",
            f"Final validation loss (step {C('train/f5_stageA.py', 'STEPS', '{:,}')}) per training seed, `true` against "
            f"`shuffled`; lines join the two runs of a seed. Validation split (used for selection); descriptive, not "
            f"the pre-registered test; {J(REPORT, 'summary/mlp/r2/n')} training seeds per arm."),
        "stimulus_eye_images_rot_left_v060_seed0.png": (
            f"What the two eyes see: raw eye images during leftward rotation at {sp1} deg/s.",
            "Luminance seen by the left (top row) and right (bottom row) FlyGym eye in the grey lead-in and at three "
            f"times after the drum starts (times in the panel titles). Training data seed {stim_seed}, grating phase {stim_phase}."),
        "stimulus_t4t5_rot_left_v060_seed0.png": (
            f"FlyVis T4/T5 boundary rates during leftward rotation at {sp1} deg/s.",
            "Heat maps of the rates (Hz, colour bar) of the T4 and T5 boundary neurons over time, left and right eye, "
            f"rows grouped by subtype (T4a-d, T5a-d); the cyan line marks the drum start. Training data seed {stim_seed}."),
        "stimulus_t4t5_rot_left_v120_seed0.png": (
            f"FlyVis T4/T5 boundary rates during leftward rotation at {sp2} deg/s.",
            f"As Figure S9. Training data seed {stim_seed}."),
        "stimulus_t4t5_rot_right_v060_seed0.png": (
            f"FlyVis T4/T5 boundary rates during rightward rotation at {sp1} deg/s.",
            f"As Figure S9. Training data seed {stim_seed}."),
        "video_still_act1_+0300ms.png": (
            f"Act 1 of the circuit video at +{st[0]} ms: change from the pre-stimulus baseline during rightward "
            f"rotation.",
            f"Each drawn neuron is coloured by its change from its own baseline (colour bars: FlyWire DNs in Hz, BANC "
            f"neurons in model units); the inset trace is the turn command dA; dashed, the teacher dA*. "
            f"{C('scripts/viz_video.py', 'SPEED')} deg/s, mean over the validation data seeds {v_lo}-{v_hi}."),
        "video_still_act2_+0300ms.png": (
            f"Act 2 of the circuit video at +{st[1]} ms: direction preference, rightward minus leftward rotation.",
            f"Orange prefers rightward, blue leftward (colour bars as in Figure S12); the inset shows dA for both "
            f"directions. {C('scripts/viz_video.py', 'SPEED')} deg/s, mean over the validation data seeds "
            f"{v_lo}-{v_hi}."),
        "video_still_act2_+0800ms.png": (
            f"Act 2 of the circuit video at +{st[2]} ms.",
            "As Figure S13, later in the trial."),
        "alignment_positions_affine_by_source.png": (
            "FlyWire brain-model neurons mapped into BANC space by our affine transform (a visualisation scaffold).",
            f"Front (x-z), side (y-z) and top (x-y) projections in BANC coordinates (µm); brain neurons coloured by the "
            f"source of their position (soma, synapse centroid, fallback centroid), the BANC VNC subgraph in light grey, "
            f"landmarks as crosses (BANC mean soma) and circles (transformed FlyWire mean soma). The title gives the "
            f"landmark fit of the affine in `docs/figures/transforms.json`: RMSE "
            f"{J(TRANSFORMS, f'{tf}/rmse_control_um', '{:.1f}')} µm on the control points and "
            f"{J(TRANSFORMS, f'{tf}/rmse_check_um', '{:.1f}')} µm on the check points, determinant "
            f"{J(TRANSFORMS, f'{tf}/determinant', '{:.3f}')}. The comparison with the BANC team's warp is in "
            f"`docs/figures/warp_check.json`."),
    }
    assert sel and set(cap) == set(files)
    assert [s[1] for s in stills] == [int(x.split("_+")[1][:4]) for x in sorted(f for f in files if f.startswith("video_"))]
    L = ["# Supporting figures", "",
         "Every file of `docs/figures/supplementary/`, by group. Sources, scripts and hashes: "
         "[figures/README.md](figures/README.md). Figures 1-4 are in the [README](../README.md).", ""]
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    for p in (README, GALLERY, NUMBERS):
        if p.exists() and not a.overwrite:
            raise SystemExit(f"{p} exists; pass --overwrite to replace it")
    texts = {"README.md": readme(), "docs/FIGURES.md": gallery()}
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
    README.write_text(texts["README.md"])
    GALLERY.write_text(texts["docs/FIGURES.md"])
    NUMBERS.write_text(json.dumps(dict(
        note="every number, name with a version, reference and command printed in README.md and docs/FIGURES.md, with "
             "its source; generated by scripts/readme_text.py", entries=nums), indent=1, ensure_ascii=False) + "\n")
    print(f"{len(nums)} entries; README {len(texts['README.md']):,} chars, gallery {len(texts['docs/FIGURES.md']):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
