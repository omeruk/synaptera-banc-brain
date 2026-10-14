"""Side-convention audit of prereg/13_side_audit.md (data and literature only; no network or flight simulation).

    env -u PYTHONPATH python scripts/side_audit.py [--codex-dir ~/Downloads] [--overwrite]

Writes docs/results/side_audit.{json,md}. When the owner's side rule (section 1: a motor neuron's side is the side of
the peripheral nerve its axon leaves through) relabels a motor neuron of a readout type, it also writes
docs/results/side_audit_post_audit.{json,md}. These recompute the docs/08 v2 prediction (`lateral`, G1) from the
existing F3b-3 and mv2 R records with the relabelled sides (section 3).

Computations:
- table look-ups on the BANC / FlyWire annotation files;
- a MuJoCo pose query of the FlyGym body (mj_forward, no time step) for link L1;
- one evaluation of the stroke-averaged aerodynamic wrench for link L8;
- re-analysis of existing run records.
"""
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

BANC = ROOT / "banc"
META = BANC / "banc_888_meta.feather"
CODEX_BANC = BANC / "neurons-banc.csv.gz"
FW_ANN = ROOT / "brain_model" / "flywire_annotations.tsv"
VB_TABLE = ROOT / "data" / "vision_boundary_783.csv"
STIM = ROOT / "runs" / "f3b3" / "stim_rates.npz"
F3B3 = ROOT / "runs" / "f3b3" / "f3b3_v3.json"
MV2_R = ROOT / "runs" / "mv2" / "t3_R.json"
F4_V1 = ROOT / "docs" / "results" / "f4_design_numbers.json"
RESULTS = ROOT / "docs" / "results"
OUT = RESULTS / "side_audit"
OUT_POST = RESULTS / "side_audit_post_audit"

DETAIL_CLASSES = ("wing_motor_neuron", "haltere_motor_neuron", "neck_motor_neuron")
DETAIL_EXTRA_TYPES = ("TTMn", "PSI")
L4_MIN, L6_MIN = 0.99, 0.95          # prereg/13 section 2
T45_FTB, T45_BTF = ("T4a", "T5a"), ("T4b", "T5b")     # front-to-back / back-to-front (Maisak et al. 2013)

# Literature cross-check (prereg/13 section 1). Nerve of each wing MN type: Azevedo et al. 2024, Nature,
# Supplementary Table 2 ("FANC links to wing motor neurons"; iv1-iv4 = hg1-hg4, "tp & tp2" = tpn, "middle leg
# tergotrochanter" = TTMn).
NERVE_NAMES = {"ADMN": "anterior_dorsal_mesothoracic_nerve", "mesoAN": "mesothoracic_accessory_nerve",
               "PDMN": "posterior_dorsal_mesothoracic_nerve"}
AZEVEDO_NERVE = {"DLM1-4": "PDMN", "DLM5": "PDMN", "DVM1a-c": "ADMN", "DVM2a-b": "mesoAN", "DVM3a-b": "PDMN",
                 "i1": "ADMN", "i2": "ADMN", "iii1": "mesoAN", "iii3": "mesoAN", "iii4": "mesoAN",
                 "hg1": "ADMN", "hg2": "ADMN", "hg3": "ADMN", "hg4": "mesoAN", "b1": "ADMN", "b2": "ADMN",
                 "b3": "ADMN", "tp1": "ADMN", "tp2": "ADMN", "tpn": "ADMN", "ps1": "mesoAN", "ps2": "mesoAN",
                 "TTMn": "PDMN"}
AZEVEDO_SOURCE = "Azevedo et al. 2024, Nature 631:360, Supplementary Table 2"
# Soma side relative to the muscle: DLM MN1-4 ipsilateral, MN5 contralateral (Ikeda & Koenig 1988, J Comp Neurol
# 273:436, as summarised by FlyBase / Virtual Fly Brain FBbt_00004079; the 1988 paper itself was not read).
SOMA_LIT = {"DLM1-4": "ipsilateral", "DLM5": "contralateral"}
SOMA_LIT_SOURCE = "Ikeda & Koenig 1988 (J Comp Neurol 273:436), as summarised by FlyBase FBbt_00004079"
LINDSAY_QUOTES = (
    ("Fig. 1B legend", "The steering muscles on the right side of the fly are imaged"),
    ("Results", "Saccades toward the ipsilateral side—when the wing stroke amplitude on the imaged side decreased"),
)
CODEX_FAQ = ("FlyWire Codex FAQ (codex.flywire.ai/faq): neuron side is decided by soma position; neuropil and neuron "
             "side annotations are from the fly's perspective; the FAFB images were left/right inverted during "
             "acquisition and all annotations and labels in CAVE and Codex were inverted to correct this.")


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    p = Path(p)
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


# ── pure helpers (tests/test_side_audit.py) ─────────────────────────────────────────────────────────────────────
def nerve_side(nerve):
    """'left' / 'right' from a BANC nerve name with a side prefix; None otherwise (no side, missing)."""
    if not isinstance(nerve, str):
        return None
    m = re.match(r"^(left|right)_", nerve)
    return m.group(1) if m else None


def neuron_verdict(soma_side, n_side):
    if n_side is None:
        return "undetermined"
    return "match" if n_side == soma_side else "relabelled"


def type_verdict(verdicts):
    v = list(verdicts)
    if not v or all(x == "undetermined" for x in v):
        return "undetermined"
    d = [x for x in v if x != "undetermined"]
    if all(x == "match" for x in d):
        return "consistent" if len(d) == len(v) else "consistent_partial"
    if all(x == "relabelled" for x in d) and len(d) == len(v):
        return "swapped"
    return "mixed"


def relabel_rates(rate_hz, verdict_by_type):
    """Readout rates keyed 'type|side' (soma side) -> nerve side. swapped: keys exchanged; mixed / partial: removed
    (per-neuron rates are not recorded). Returns (new dict, removed types)."""
    out, removed = {}, set()
    for k, v in rate_hz.items():
        t, _, s = k.rpartition("|")
        vd = verdict_by_type.get(t, "consistent")
        if vd == "swapped" and s in ("left", "right"):
            out[f"{t}|{'right' if s == 'left' else 'left'}"] = v
        elif vd in ("mixed", "consistent_partial", "undetermined") and t in verdict_by_type:
            removed.add(t)
        else:
            out[k] = v
    return out, sorted(removed)


# ── section 1: motor neurons ────────────────────────────────────────────────────────────────────────────────────
def load_banc_tables():
    from connectome.adapters.banc import load_neurons
    n = load_neurons(CODEX_BANC)
    meta = pd.read_feather(META, columns=["banc_888_id", "side", "nerve", "super_class", "cell_class", "cell_type",
                                          "fafb_match", "fafb_nblast_match", "root_position"])
    meta["id"] = meta["banc_888_id"].astype(np.int64)
    return n, meta.drop(columns="banc_888_id")


def motor_table(n, meta):
    """One row per BANC motor neuron (meta super_class motor), model columns from the Codex table."""
    mm = meta[meta["super_class"] == "motor"].rename(columns={"side": "meta_side", "nerve": "meta_nerve",
                                                             "cell_class": "meta_class", "cell_type": "meta_type"})
    t = mm.merge(n[["id", "cell_type", "cell_class", "super_class", "sub_class", "side", "nerve"]], on="id", how="left",
                 suffixes=("_meta", ""))
    t["in_codex"] = t["cell_type"].notna()
    t["type"] = t["cell_type"].where(t["in_codex"], t["meta_type"]).astype(str)
    t["class"] = t["cell_class"].where(t["in_codex"], t["meta_class"]).astype(str)
    t["soma_side"] = t["side"].where(t["in_codex"], t["meta_side"]).astype(str)
    t["nerve_side"] = [nerve_side(x) for x in t["meta_nerve"]]
    t["verdict"] = [neuron_verdict(s, ns) for s, ns in zip(t["soma_side"], t["nerve_side"])]
    t["rule_side"] = [ns if ns is not None else s for s, ns in zip(t["soma_side"], t["nerve_side"])]
    return t.sort_values(["class", "type", "soma_side", "id"]).reset_index(drop=True)


def readout_type_names(n):
    import f3a_vnc_run as F
    return sorted(F.readout_types(n))


def section1(n, meta):
    t = motor_table(n, meta)
    readout = set(readout_type_names(n))
    det = t[t["class"].isin(DETAIL_CLASSES) | t["type"].isin(DETAIL_EXTRA_TYPES)]
    neurons = [dict(id=int(r.id), type=r.type, cell_class=r["class"], soma_side=r.soma_side,
                    meta_side=str(r.meta_side), nerve=str(r.meta_nerve), codex_nerve=str(r.nerve),
                    nerve_side=r.nerve_side, verdict=r.verdict, root_position=str(r.root_position))
               for _, r in det.iterrows()]
    types = {}
    for (cls, typ), g in det.groupby(["class", "type"], sort=True):
        vd = type_verdict(g["verdict"])
        nerves = sorted({str(x).removeprefix("left_").removeprefix("right_") for x in g["meta_nerve"]})
        lit_n = AZEVEDO_NERVE.get(typ)
        types[typ] = dict(
            cell_class=cls, n=int(len(g)),
            n_soma_left=int((g["soma_side"] == "left").sum()), n_soma_right=int((g["soma_side"] == "right").sum()),
            n_nerve_left=int((g["nerve_side"] == "left").sum()), n_nerve_right=int((g["nerve_side"] == "right").sum()),
            n_match=int((g["verdict"] == "match").sum()), n_relabelled=int((g["verdict"] == "relabelled").sum()),
            n_undetermined=int((g["verdict"] == "undetermined").sum()), verdict=vd, nerves=nerves,
            readout=typ in readout,
            lit_nerve=(dict(abbrev=lit_n, name=NERVE_NAMES[lit_n], agrees=nerves == [NERVE_NAMES[lit_n]])
                       if lit_n else None),
            lit_soma=SOMA_LIT.get(typ),
            lit_soma_agrees=(None if typ not in SOMA_LIT else
                             (vd == "consistent") == (SOMA_LIT[typ] == "ipsilateral")))
    by_class = {}
    for cls, g in t.groupby("class", sort=True):
        tv = {typ: type_verdict(gg["verdict"]) for typ, gg in g.groupby("type", sort=True)}
        by_class[cls] = dict(n=int(len(g)), n_sided_nerve=int(g["nerve_side"].notna().sum()),
                             n_match=int((g["verdict"] == "match").sum()),
                             n_relabelled=int((g["verdict"] == "relabelled").sum()),
                             n_undetermined=int((g["verdict"] == "undetermined").sum()),
                             relabelled_types=sorted(k for k, v in tv.items() if v in ("swapped", "mixed")),
                             undetermined_nerves=sorted({str(x) for x in g.loc[g["nerve_side"].isna(), "meta_nerve"]}))
    all_types = {typ: type_verdict(g["verdict"]) for typ, g in t.groupby("type", sort=True)}
    relab = sorted(k for k, v in all_types.items() if v in ("swapped", "mixed"))
    return dict(n_motor=int(len(t)), n_not_in_codex=int((~t["in_codex"]).sum()),
                n_codex_meta_side_differ=int((t["in_codex"] & (t["side"].astype(str) != t["meta_side"].astype(str))).sum()),
                n_codex_meta_nerve_differ=int((t["in_codex"] & (t["nerve"].astype(str) != t["meta_nerve"].astype(str))).sum()),
                types=types, neurons=neurons, by_class=by_class, relabelled_types=relab,
                relabelled_readout_types=sorted(k for k in relab if k in readout),
                verdict_all_types={k: v for k, v in all_types.items() if v != "consistent"})


# ── section 2: the chain ────────────────────────────────────────────────────────────────────────────────────────
def link_l1_l2():
    """Grating phase direction (pure formula) and the FlyGym body frame (pose query, no time step)."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    from body.flygym_body import FlightBody
    from tasks import t0_stimuli as S
    # constant-phase point of condition 1: az / lambda + TF t = const -> d az / dt = -lambda TF
    az = np.linspace(-170, 170, 69)
    rays = np.stack([np.cos(np.radians(az)), np.sin(np.radians(az)), np.zeros_like(az)], -1)
    dt = 1e-3
    l0, l1 = S.luminance(1, 0.2, rays), S.luminance(1, 0.2 + dt, rays)
    # the pattern moves by v: l(az, t + dt) = l(az - v dt, t); compare to a shifted copy
    v_expected = -S.WAVELENGTH_DEG * S.TF_HZ
    l_shift = S.luminance(1, 0.2, np.stack([np.cos(np.radians(az - v_expected * dt)),
                                             np.sin(np.radians(az - v_expected * dt)), np.zeros_like(az)], -1))
    phase_ok = bool(np.allclose(l1, l_shift, atol=1e-9)) and not np.allclose(l0, l1)
    l_shift_wrong = S.luminance(1, 0.2, np.stack([np.cos(np.radians(az + v_expected * dt)),
                                                   np.sin(np.radians(az + v_expected * dt)), np.zeros_like(az)], -1))
    b = FlightBody(spawn_pos=(0, 0, 100), spawn_yaw=0.0, legs="tuck", enable_vision=True)
    mujoco.mj_forward(b.m, b.d)
    Rb, xb = b.d.xmat[b.root].reshape(3, 3), b.d.xpos[b.root]
    yb = {}
    for k in range(b.m.nbody):
        nm = mujoco.mj_id2name(b.m, mujoco.mjtObj.mjOBJ_BODY, k) or ""
        base = nm.split("/")[-1]
        if base in ("LEye", "REye", "LWing", "RWing") or base.endswith("Coxa"):
            yb[base] = float((Rb.T @ (b.d.xpos[k] - xb))[1])
    for k in range(b.m.ncam):
        nm = mujoco.mj_id2name(b.m, mujoco.mjtObj.mjOBJ_CAMERA, k).split("/")[-1]
        yb[nm] = float((Rb.T @ (b.d.cam_xpos[k] - xb))[1])
    # prereg/13 criterion: the eye cameras and the coxae (3 per side; the prereg's "six L* coxae" means the six
    # coxae, three per side). The eye and wing bodies are reported only (the eye body origins lie on the midline).
    crit = [k for k in yb if k.endswith("Eye_cam") or k.endswith("Coxa")]
    left_ok = all(yb[k] > 0 for k in crit if k.startswith("L"))
    right_ok = all(yb[k] < 0 for k in crit if k.startswith("R"))
    rays_eye = S.camera_rays(b.m, b.d, b.root)
    az_c = [float(S.azimuth(rays_eye[e, S.H_PX // 2, S.W_PX // 2])) for e in (0, 1)]
    l1v = "OK" if (phase_ok and left_ok and right_ok) else (
        "SWAPPED" if (np.allclose(l1, l_shift_wrong, atol=1e-9) and not left_ok and not right_ok) else "UNRESOLVED")
    l2_code = "camera_rays iterates ('LEye_cam', 'REye_cam') -> eye 0 = left; FlyVis batch row 0 -> group 'L' (sense/vision_boundary.py BoundaryEyes)"
    l2v = "OK" if az_c[0] > 0 > az_c[1] else ("SWAPPED" if az_c[0] < 0 < az_c[1] else "UNRESOLVED")
    return (dict(verdict=l1v, rot_right_phase_velocity_deg_s=float(v_expected), phase_moves_to_negative_azimuth=phase_ok,
                 body_frame_y_mm=yb, criterion_bodies=sorted(crit), left_named_y_positive=left_ok, right_named_y_negative=right_ok,
                 convention="body frame x forward, y left, z up; azimuth + = left (tasks/t0_stimuli.py)"),
            dict(verdict=l2v, centre_ray_azimuth_deg=dict(eye0=az_c[0], eye1=az_c[1]), code=l2_code))


def link_l3_l4(codex_dir):
    from connectome.adapters import flywire783 as fw
    import f3b1_common as C1
    ids = fw.load_root_ids()
    pos = pd.Series(np.arange(len(ids)), index=ids.astype(np.int64))
    tab = pd.read_csv(VB_TABLE)
    tab = tab[tab["root_id"].isin(pos.index)].copy()
    tab["eng"] = pos.loc[tab["root_id"].to_numpy()].to_numpy()
    z = np.load(STIM)
    lo, hi = C1.WINDOW
    contrast, groups = {}, {}
    for h, s in (("left", "L"), ("right", "R")):
        g = tab[tab["hemisphere"] == h].sort_values("eng")
        groups[s] = g
        ft = g["fw_type"].to_numpy()
        for st in ("rot_right", "rot_left", "static", "grey"):
            r = z[f"{st}__{s}"]
            if r.shape[1] != len(g):
                raise SystemExit(f"stim_rates {st}__{s}: {r.shape[1]} columns, boundary group has {len(g)}")
            m = r[lo:hi].mean(0)
            contrast[f"{st}|{h}"] = float(np.mean([m[ft == k].mean() for k in T45_FTB]) -
                                          np.mean([m[ft == k].mean() for k in T45_BTF]))
    exp = {"rot_right|right": 1, "rot_right|left": -1, "rot_left|left": 1, "rot_left|right": -1}
    signs = [np.sign(contrast[k]) == v for k, v in exp.items()]
    l3_groups_ok = bool((groups["L"]["hemisphere"] == "left").all() and (groups["R"]["hemisphere"] == "right").all())
    l3v = "OK" if (all(signs) and l3_groups_ok) else ("SWAPPED" if not any(signs) else "UNRESOLVED")
    ann = pd.read_csv(FW_ANN, sep="\t", usecols=["root_id", "side"])
    ann["root_id"] = ann["root_id"].astype(np.int64)
    j = tab.merge(ann, on="root_id", how="inner")
    agree_bnd = float((j["hemisphere"] == j["side"]).mean())
    codex = Path(codex_dir).expanduser() / "classification.csv.gz"
    cls = None
    if codex.exists():
        c = pd.read_csv(codex, usecols=["root_id", "side"])
        jj = ann.merge(c, on="root_id", suffixes=("_ann", "_codex"))
        cls = dict(file=codex.name, sha256=sha256(codex), n_shared=int(len(jj)),
                   agreement=float((jj["side_ann"] == jj["side_codex"]).mean()),
                   crosstab={f"{a}|{b}": int(v) for (a, b), v in
                             jj.groupby(["side_ann", "side_codex"]).size().items()})
    if cls is None:
        l4v = "UNRESOLVED"
    elif agree_bnd >= L4_MIN and cls["agreement"] >= L4_MIN:
        l4v = "OK"
    elif agree_bnd <= 1 - L4_MIN or cls["agreement"] <= 1 - L4_MIN:
        l4v = "SWAPPED"
    else:
        l4v = "UNRESOLVED"
    return (dict(verdict=l3v, ftb_minus_btf_hz=contrast, expected_sign=exp, groups_match_hemisphere=l3_groups_ok,
                 n_L=int(len(groups["L"])), n_R=int(len(groups["R"])), window=[int(lo), int(hi)],
                 note="(T4a+T5a)-(T4b+T5b) window mean rate per eye; T4a/T5a prefer front-to-back motion"),
            dict(verdict=l4v, boundary_n=int(len(j)), boundary_hemisphere_equals_annotation_side=agree_bnd,
                 annotation_vs_codex=cls, documentation=CODEX_FAQ))


def link_l5_l6(n, meta):
    import f3b2_common as C2
    from coupling.dn_bridge import BridgeConfig, DNBridge, family
    z = np.load(sorted((ROOT / "runs" / "f3b3" / "brain").glob("rot_left_s*.npz"))[0], allow_pickle=True)
    fw = pd.DataFrame(dict(cell_type=z["dn_cell_type"].astype(str), side=z["dn_side"].astype(str)))
    dn = n[n["super_class"] == "descending"].reset_index(drop=True)
    banc = pd.DataFrame(dict(cell_type=dn["cell_type"].astype(str), side=dn["side"].astype(str),
                             index=np.arange(len(dn))))
    b = DNBridge(fw, banc, BridgeConfig(families=C2.ENSEMBLE, latency_intervals=1, window_intervals=1))
    tgt_side = banc.set_index("index").loc[b.target_index, "side"].to_numpy()
    res = {}
    for drive in ("left", "right"):
        c = np.zeros((3, len(fw)))
        c[0, (fw["side"] == drive).to_numpy()] = 1.0
        r = b.rates(c)[1]
        res[drive] = dict(n_targets_driven=int((r > 0).sum()), n_driven_same_side=int(((r > 0) & (tgt_side == drive)).sum()),
                          n_driven_other_side=int(((r > 0) & (tgt_side != drive)).sum()),
                          n_targets_on_side=int((tgt_side == drive).sum()))
    same = all(v["n_driven_other_side"] == 0 and v["n_driven_same_side"] == v["n_targets_on_side"] for v in res.values())
    other = all(v["n_driven_same_side"] == 0 for v in res.values())
    l5 = dict(verdict="OK" if same else ("SWAPPED" if other else "UNRESOLVED"), per_drive=res,
              n_units=len(b.units), families=len(C2.ENSEMBLE),
              note="real F3b-3 FlyWire DN table and the BANC Codex DN table; one interval of unit counts on one side")
    md = meta[meta["super_class"] == "descending"][["id", "side", "cell_type", "fafb_match", "fafb_nblast_match"]]
    md = md.merge(dn[["id", "side", "cell_type"]].rename(columns={"side": "codex_side", "cell_type": "codex_type"}),
                  on="id", how="inner")
    ann = pd.read_csv(FW_ANN, sep="\t", usecols=["root_id", "side"])
    ann = ann.assign(root_id=ann["root_id"].astype(str)).set_index("root_id")["side"]
    out = {}
    for col in ("fafb_match", "fafb_nblast_match"):
        fid = md[col].astype(str).str.split(",").str[0].str.strip()
        fs = fid.map(ann)
        ok = fs.isin(["left", "right"]) & md["codex_side"].isin(["left", "right"])
        ens = md["codex_type"].map(family).isin(C2.ENSEMBLE)
        sub = {}
        for name, m in (("all", ok), ("ensemble", ok & ens)):
            sub[name] = dict(n=int(m.sum()), n_same=int((m & (fs == md["codex_side"])).sum()),
                             agreement=float((fs[m] == md["codex_side"][m]).mean()) if m.any() else None)
        out[col] = sub
    a = [out[c][k]["agreement"] for c in ("fafb_match",) for k in ("all", "ensemble")]
    l6v = "OK" if all(x is not None and x >= L6_MIN for x in a) else (
        "SWAPPED" if all(x is not None and x <= 1 - L6_MIN for x in a) else "UNRESOLVED")
    l6 = dict(verdict=l6v, match=out, criterion_column="fafb_match", n_banc_dn=int(len(md)),
              note="BANC side = Codex Soma side (the side the model uses); FlyWire side = flywire_annotations.tsv side "
                   "of the first listed match; neurons with a center / unknown side on either side are left out. "
                   "fafb_nblast_match is reported, not a criterion.")
    return l5, l6


def link_l7(s1):
    lat = ("b2", "b3", "i1", "i2", "hg1", "hg2", "hg3", "hg4")
    rows = {t: s1["types"][t]["verdict"] for t in lat}
    ok = all(v == "consistent" for v in rows.values())
    sw = all(v == "swapped" for v in rows.values())
    return dict(verdict="OK" if ok else ("SWAPPED" if sw else "UNRESOLVED"), mapping_types=rows,
                note="DN and MN sides are the same Codex Soma side column of one dataset; the rule of section 1 decides "
                     "whether an MN's soma side is its muscle side")


def link_l8_check():
    from body import quasi_steady as qs
    import inspect
    src = inspect.getsource(qs.turn_to_amplitudes)
    aL, aR = qs.turn_to_amplitudes(1.0, 140.0, 10.0)
    model_right_turn_larger_left = bool(aL > aR and "turn right" in src)
    # Lindsay: a saccade toward the ipsilateral (imaged) side goes with a smaller ipsilateral amplitude, i.e. a turn
    # toward the side with the smaller amplitude: right turn <-> A_R < A_L. Same as the model.
    lindsay_right_turn_larger_left = True
    ok = model_right_turn_larger_left == lindsay_right_turn_larger_left
    return dict(verdict="OK" if ok else "SWAPPED", quotes=[dict(location=a, text=b) for a, b in LINDSAY_QUOTES],
                model_turn_convention="turn_to_amplitudes(turn_bias > 0) -> A_L > A_R, docstring 'turn right'; "
                                      "tests/test_arena_wing.py::test_turn_sign_in_wrench: yaw torque < 0 (right)",
                model_right_turn_larger_left=model_right_turn_larger_left,
                lindsay_right_turn_larger_left=lindsay_right_turn_larger_left,
                mapping="A_side = A0 + G sum_m s_m r_{m,side}; s_m = effect on the ipsilateral (same-side) amplitude; "
                        "dA = A_L - A_R > 0 turns right; S > 0 = syndirectional (scripts/f4_design_numbers.py)",
                imaged_side_matters=False,
                imaged_side_note="Lindsay imaged the right side; the signs are stated for the ipsilateral wing, so they "
                                 "apply to both sides under mirror symmetry")


# ── section 3: post-audit recomputation ─────────────────────────────────────────────────────────────────────────
def recompute(path, verdict_by_type, relabel):
    import f4_design_numbers as F4
    import mv2_report as R
    d = json.loads(Path(path).read_text())
    removed = set()
    if relabel:
        for r in d["replay"]:
            r["rate_hz"], rm = relabel_rates(r["rate_hz"], verdict_by_type)
            removed |= set(rm)
    rates, seeds = F4.load_rates(d)
    signs = F4.SIGNS_LATERAL_V2
    if set(signs) & removed:
        return dict(recomputable=False, removed_types=sorted(removed))
    g1 = json.loads(F4_V1.read_text())["gain"]["G1_deg_per_hz"]
    arms = list(d["constants"]["arms"])
    randoms = [a for a in arms if a.startswith("random_")]
    s = {a: F4.s_index(rates, a, seeds, signs) for a in arms}
    e = s["ensemble"]
    n_ge = int(sum(abs(s[a].mean()) >= abs(e.mean()) for a in randoms))
    types = [m for m in d["constants"]["steer_mn"] if d["symmetry"].get(m, {}).get("passed")]
    D = F4.d_per_type(rates, "ensemble", seeds, types)
    per_type = {m: dict(T=float(D[m].mean()), sd=float(D[m].std(ddof=1)), p_two_sided=R.p_t(D[m]),
                        contribution_to_S=(float(-signs[m] * D[m].mean() / 2.0) if m in signs else None))
                for m in types}
    return dict(recomputable=True, removed_types=sorted(removed), seeds=seeds, n_random=len(randoms),
                S_mean=float(e.mean()), S_sd=float(e.std(ddof=1)), S_values=[float(x) for x in e],
                p_one_sided=R.p_one_sided(e), p_two_sided=R.p_t(e), n_pos=int((e > 0).sum()), n_neg=int((e < 0).sum()),
                bridge_off_S_mean=float(s["bridge_off"].mean()), bridge_off_p_two_sided=R.p_t(s["bridge_off"]),
                random_S_means=[float(s[a].mean()) for a in randoms], n_random_abs_ge=n_ge,
                perm_p=(1 + n_ge) / (len(randoms) + 1), turn_deg_G1=float(g1 * e.mean()), G1_deg_per_hz=g1,
                per_type=per_type)


def section3(s1):
    vb = {t: s1["types"][t]["verdict"] for t in s1["types"]}
    vb.update(s1["verdict_all_types"])
    rec = {}
    for name, p in (("f3b3", F3B3), ("mv2_R", MV2_R)):
        pre = recompute(p, vb, relabel=False)
        post = recompute(p, vb, relabel=True)
        same = post.get("recomputable") and all(
            json.dumps(pre[k], sort_keys=True) == json.dumps(post[k], sort_keys=True)
            for k in pre if k not in ("removed_types",))
        rec[name] = dict(record=rel(p), sha256=sha256(p), pre_audit=pre, post_audit=post, identical=bool(same))
    in_records = set()
    for p in (F3B3, MV2_R):
        d = json.loads(p.read_text())
        in_records |= {k.rpartition("|")[0] for r in d["replay"] for k in r["rate_hz"]}
    return dict(label="post-audit", signs=dict(__import__("f4_design_numbers").SIGNS_LATERAL_V2), records=rec,
                swapped_types=sorted(t for t, v in vb.items() if v == "swapped" and t in in_records),
                not_recomputable_types=sorted(t for t, v in vb.items() if v == "mixed" and t in in_records))


# ── section 4: earlier results that contain a relabelled type ──────────────────────────────────────────────────
def _walk(o, path, t, hits):
    if isinstance(o, dict):
        for k, v in o.items():
            ks = str(k)
            if ks == t or ks.startswith(f"{t}|") or ks.endswith(f"|{t}"):
                hits.append(f"{path}/{ks}")
            _walk(v, f"{path}/{ks}", t, hits)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            _walk(v, f"{path}[{i}]", t, hits)
    elif isinstance(o, str) and (o == t or o.startswith(f"{t}|")):
        hits.append(path)


def section4(types, own=("side_audit.json", "side_audit_post_audit.json")):
    out = {}
    for t in types:
        files = {}
        for p in sorted(RESULTS.glob("*.json")):
            if p.name in own:
                continue
            try:
                d = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            hits = []
            _walk(d, "", t, hits)
            if hits:
                files[p.name] = dict(n=len(hits), examples=sorted(set(re.sub(r"\[\d+\]", "[*]", h) for h in hits))[:4])
        for p in sorted(RESULTS.glob("*.md")):
            if p.name.startswith("side_audit"):
                continue
            k = len(re.findall(rf"(?<![\w-]){re.escape(t)}(?![\w-])", p.read_text()))
            if k:
                files.setdefault(p.name, {})["md_mentions"] = k
        out[t] = files
    return out


# ── build / render ──────────────────────────────────────────────────────────────────────────────────────────────
def build(codex_dir="~/Downloads"):
    n, meta = load_banc_tables()
    s1 = section1(n, meta)
    l1, l2 = link_l1_l2()
    l3, l4 = link_l3_l4(codex_dir)
    l5, l6 = link_l5_l6(n, meta)
    l7 = link_l7(s1)
    l8 = link_l8_check()
    chain = {"L1 rot_right -> eye images": l1, "L2 eye images -> FlyVis eye": l2,
             "L3 FlyVis eye -> boundary side": l3, "L4 boundary side -> FlyWire side": l4,
             "L5 FlyWire side -> bridge": l5, "L6 bridge -> BANC DN side": l6,
             "L7 BANC DN side -> BANC MN side": l7, "L8 BANC MN side -> mapping side": l8}
    triggered = bool(s1["relabelled_readout_types"])
    o = dict(prereg="prereg/13_side_audit.md",
             inputs={rel(p): sha256(p) for p in (META, CODEX_BANC, FW_ANN, VB_TABLE, STIM)},
             literature=dict(nerve=AZEVEDO_SOURCE, soma=SOMA_LIT_SOURCE, flywire_side=CODEX_FAQ,
                             mapping="Lindsay, Sustar & Dickinson 2017, Curr Biol 27:345"),
             section1=s1, chain=chain, chain_verdicts={k: v["verdict"] for k, v in chain.items()},
             post_audit_triggered=triggered,
             earlier_results=section4(s1["relabelled_types"]))
    post = section3(s1) if triggered else None
    return o, post


def f(x, n=2, sign=False):
    return "-" if x is None else (f"{x:+.{n}f}" if sign else f"{x:.{n}f}")


def pf(p):
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def render(o):
    s1 = o["section1"]
    L = ["# Side-convention audit (prereg/13)", "",
         "Data and literature only; no network or flight simulation. Generated by `scripts/side_audit.py`; checked by "
         "`scripts/verify_report.py`.", "",
         "## 1. Motor neurons: soma side against the side of the exit nerve", "",
         f"Rule (owner): a motor neuron's side is the side of the peripheral nerve its axon leaves through (BANC v888 meta "
         f"`nerve`). The model's side is the Codex `Soma side`. BANC motor neurons: {s1['n_motor']}; not in the Codex "
         f"table: {s1['n_not_in_codex']}; Codex and meta side differ: {s1['n_codex_meta_side_differ']}; Codex and meta "
         f"nerve differ: {s1['n_codex_meta_nerve_differ']}.", "",
         "### 1.1 Wing, haltere and neck MN types (and TTMn)", "",
         "| class | type | n | soma L/R | nerve L/R | relabelled | verdict | nerve(s) | literature nerve | literature soma | readout |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, r in sorted(s1["types"].items(), key=lambda kv: (kv[1]["cell_class"], kv[0])):
        ln = r["lit_nerve"]
        lit = "-" if ln is None else f"{ln['abbrev']} ({'agrees' if ln['agrees'] else 'DIFFERS'})"
        ls = "-" if r["lit_soma"] is None else f"{r['lit_soma']} ({'agrees' if r['lit_soma_agrees'] else 'DIFFERS'})"
        L.append(f"| {r['cell_class'].replace('_motor_neuron', '')} | {t} | {r['n']} | {r['n_soma_left']}/{r['n_soma_right']} | "
                 f"{r['n_nerve_left']}/{r['n_nerve_right']} | {r['n_relabelled']} | {r['verdict']} | "
                 f"{', '.join(r['nerves'])} | {lit} | {ls} | {'yes' if r['readout'] else 'no'} |")
    rel_n = [x for x in s1["neurons"] if x["verdict"] == "relabelled"]
    L += ["", "Relabelled neurons:", "", "| id | type | soma side | exit nerve | root position |", "|---|---|---|---|---|"]
    L += [f"| {x['id']} | {x['type']} | {x['soma_side']} | {x['nerve']} | {x['root_position']} |" for x in rel_n]
    L += ["", "### 1.2 Every MN class", "",
          "| class | n | with a sided nerve | match | relabelled | undetermined | relabelled types | nerves without side |",
          "|---|---|---|---|---|---|---|---|"]
    for c, r in s1["by_class"].items():
        L.append(f"| {c} | {r['n']} | {r['n_sided_nerve']} | {r['n_match']} | {r['n_relabelled']} | {r['n_undetermined']} | "
                 f"{', '.join(r['relabelled_types']) or '-'} | {', '.join(r['undetermined_nerves']) or '-'} |")
    L += ["", f"Relabelled types (all classes): {', '.join(s1['relabelled_types']) or 'none'}. "
              f"Of these, readout types of the model: {', '.join(s1['relabelled_readout_types']) or 'none'}.", "",
          "Literature: nerves from " + o["literature"]["nerve"] + "; soma sides from " + o["literature"]["soma"] +
          ". No primary source on the soma side of the direct (steering) wing, haltere or neck MNs was read; for them "
          "the BANC nerve annotation is the evidence.", "",
          "## 2. The left/right chain", "", "| link | verdict |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in o["chain_verdicts"].items()]
    c = o["chain"]
    l1, l2, l3, l4 = (c["L1 rot_right -> eye images"], c["L2 eye images -> FlyVis eye"],
                      c["L3 FlyVis eye -> boundary side"], c["L4 boundary side -> FlyWire side"])
    l5, l6, l7, l8 = (c["L5 FlyWire side -> bridge"], c["L6 bridge -> BANC DN side"],
                      c["L7 BANC DN side -> BANC MN side"], c["L8 BANC MN side -> mapping side"])
    yb = l1["body_frame_y_mm"]
    L += ["",
          f"- **L1.** A constant-phase point of `rot_right` moves at {f(l1['rot_right_phase_velocity_deg_s'], 1, True)} deg/s "
          f"in azimuth (+ = left), i.e. to the fly's right ({'yes' if l1['phase_moves_to_negative_azimuth'] else 'no'}). "
          f"Body-frame y (mm, + = left) of the criterion bodies named left: "
          + ", ".join(f"{k} {f(yb[k], 3, True)}" for k in l1["criterion_bodies"] if k.startswith("L"))
          + "; named right: " + ", ".join(f"{k} {f(yb[k], 3, True)}" for k in l1["criterion_bodies"] if k.startswith("R"))
          + ". Reported only: " + ", ".join(f"{k} {f(v, 3, True)}" for k, v in sorted(yb.items())
                                            if k not in l1["criterion_bodies"])
          + " (the eye body origins lie on the midline). The prereg's \"six L* coxae\" are the six coxae, three per side.",
          f"- **L2.** Centre-ray azimuth of eye 0 {f(l2['centre_ray_azimuth_deg']['eye0'], 1, True)} deg, eye 1 "
          f"{f(l2['centre_ray_azimuth_deg']['eye1'], 1, True)} deg. {l2['code']}.",
          f"- **L3.** Boundary groups: L {l3['n_L']} neurons, R {l3['n_R']}; every group matches its column-table "
          f"hemisphere: {'yes' if l3['groups_match_hemisphere'] else 'no'}. (T4a+T5a) - (T4b+T5b), window mean (Hz), F3b-3 renders:", "",
          "| stimulus | left eye | right eye |", "|---|---|---|"]
    for st in ("rot_right", "rot_left", "static", "grey"):
        L.append(f"| {st} | {f(l3['ftb_minus_btf_hz'][st + '|left'], 2, True)} | {f(l3['ftb_minus_btf_hz'][st + '|right'], 2, True)} |")
    a = l4["annotation_vs_codex"]
    L += ["", "  Expected (front-to-back motion): `rot_right` right eye > 0, left eye < 0; `rot_left` the reverse.",
          f"- **L4.** Column-table hemisphere = FlyWire annotation `side` for {f(100 * l4['boundary_hemisphere_equals_annotation_side'], 2)} % "
          f"of {l4['boundary_n']} boundary neurons. Annotation `side` = Codex `classification` `side`: "
          + ("not checked (file missing)" if a is None else f"{f(100 * a['agreement'], 3)} % of {a['n_shared']} shared neurons")
          + ". " + l4["documentation"],
          f"- **L5.** Bridge on the real DN tables ({l5['n_units']} units): drive left -> "
          f"{l5['per_drive']['left']['n_driven_same_side']} left / {l5['per_drive']['left']['n_driven_other_side']} right "
          f"targets driven (of {l5['per_drive']['left']['n_targets_on_side']} left); drive right -> "
          f"{l5['per_drive']['right']['n_driven_same_side']} right / {l5['per_drive']['right']['n_driven_other_side']} left "
          f"(of {l5['per_drive']['right']['n_targets_on_side']} right).",
          f"- **L6.** BANC DN soma side = FlyWire side of `fafb_match`: all DNs {l6['match']['fafb_match']['all']['n_same']} / "
          f"{l6['match']['fafb_match']['all']['n']} ({f(100 * l6['match']['fafb_match']['all']['agreement'], 1)} %), ensemble "
          f"{l6['match']['fafb_match']['ensemble']['n_same']} / {l6['match']['fafb_match']['ensemble']['n']} "
          f"({f(100 * l6['match']['fafb_match']['ensemble']['agreement'], 1)} %). `fafb_nblast_match` (reported only): all "
          f"{l6['match']['fafb_nblast_match']['all']['n_same']} / {l6['match']['fafb_nblast_match']['all']['n']}, ensemble "
          f"{l6['match']['fafb_nblast_match']['ensemble']['n_same']} / {l6['match']['fafb_nblast_match']['ensemble']['n']}.",
          f"- **L7.** Mapping types: " + ", ".join(f"{k} {v}" for k, v in l7["mapping_types"].items()) + ". " + l7["note"] + ".",
          f"- **L8.** " + "; ".join(f"Lindsay et al. 2017 ({q['location']}): \"{q['text']}\"" for q in l8["quotes"])
          + f". Model: {l8['model_turn_convention']}. {l8['mapping']}. {l8['imaged_side_note']}.", "",
          "Absolute chirality: a mirror image has the same connectivity, so whether \"left\" is the fly's left cannot be "
          "measured from the connectome. For FlyWire it is documented (Codex FAQ, L4). For BANC no statement was found in "
          "the methods text read for this audit; L6 shows that the two datasets' labels agree with each other.", "",
          "## 3. Post-audit recomputation", ""]
    if o["post_audit_triggered"]:
        L.append("Triggered: readout types are relabelled. See `docs/results/side_audit_post_audit.md`.")
    else:
        L.append("Not triggered: no readout type is relabelled.")
    L += ["", "## 4. Earlier results that contain a relabelled type (listed, not recomputed)", "",
          "| type | verdict | result file | JSON hits | markdown mentions | example JSON paths |", "|---|---|---|---|---|---|"]
    allv = {**s1["verdict_all_types"], **{k: v["verdict"] for k, v in s1["types"].items()}}
    for t, files in o["earlier_results"].items():
        if not files:
            L.append(f"| {t} | {allv.get(t, '-')} | none | 0 | 0 | - |")
        for fn, x in files.items():
            L.append(f"| {t} | {allv.get(t, '-')} | {fn} | {x.get('n', 0)} | {x.get('md_mentions', 0)} | "
                     f"{'; '.join('`' + e + '`' for e in x.get('examples', [])) or '-'} |")
    steer = json.loads(F3B3.read_text())["constants"]["steer_mn"]
    rs = [t for t in steer if allv.get(t, "consistent") != "consistent"]
    L += ["", "A `swapped` type exchanges its left and right keys: its symmetry-gate |log2(R/L)| is unchanged, its "
              "per-side rates and any left/right difference change sign. A `mixed` type changes its per-side counts "
              "(ADNM1 and CvN7 have both somata on the left, so they had no gate row; by nerve they are one per side).",
          "", f"Steering MN types of F3a-3 / F3b (`steer_mn`: {', '.join(steer)}) relabelled: {', '.join(rs) or 'none'}. "
              "The F3a-3 steering signs, the F3b per-type T and the docs/08 mapping therefore do not change; the types "
              "listed above enter only the power, neck, PSI and per-side rate tables."]
    return "\n".join(L) + "\n"


def render_post(p):
    L = ["# Side-convention audit: post-audit recomputation (prereg/13 section 3)", "",
         "Label: **post-audit**. Existing records re-analysed with the motor-neuron sides of the owner's rule; nothing was "
         "rerun. Mapping: docs/08 section 10 v2 `lateral` (" + ", ".join(f"{k}{'+' if v > 0 else '-'}" for k, v in p["signs"].items())
         + "), gain G1.", "",
         f"Swapped readout types (keys exchanged): {', '.join(p['swapped_types']) or 'none'}. Not recomputable (mixed; "
         f"per-neuron rates not recorded): {', '.join(p['not_recomputable_types']) or 'none'}.", "",
         "| record | version | S mean (Hz) | sd | p one-sided | p two-sided | seeds +/- | random abs >= | perm p | turn at G1 (deg) | bridge_off S (p) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in p["records"].items():
        for ver in ("pre_audit", "post_audit"):
            x = r[ver]
            if not x.get("recomputable"):
                L.append(f"| {name} | {ver} | not recomputable | | | | | | | | |")
                continue
            L.append(f"| {name} | {ver.replace('_', '-')} | {f(x['S_mean'], 2, True)} | {f(x['S_sd'])} | {pf(x['p_one_sided'])} | "
                     f"{pf(x['p_two_sided'])} | {x['n_pos']} / {x['n_neg']} | {x['n_random_abs_ge']} / {x['n_random']} | "
                     f"{f(x['perm_p'], 3)} | {f(x['turn_deg_G1'], 2, True)} | {f(x['bridge_off_S_mean'], 2, True)} ({pf(x['bridge_off_p_two_sided'])}) |")
    L += [""]
    for name, r in p["records"].items():
        L.append(f"- `{name}` ({r['record']}): post-audit identical to pre-audit: {'yes' if r['identical'] else 'NO'}.")
    L += ["", "Per-type T = mean over seeds of D_m (Hz) and contribution to S = -s_m T / 2, post-audit:", "",
          "| type | " + " | ".join(f"{n} T | {n} contribution" for n in p["records"]) + " |",
          "|---|" + "---|---|" * len(p["records"])]
    types = list(next(iter(p["records"].values()))["post_audit"].get("per_type", {}))
    for t in types:
        cells = []
        for r in p["records"].values():
            x = r["post_audit"]["per_type"].get(t)
            cells += [f(x["T"], 2, True), f(x["contribution_to_S"], 2, True)] if x else ["-", "-"]
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--codex-dir", default="~/Downloads", help="directory with the FlyWire Codex classification.csv.gz")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    outs = [OUT.with_suffix(".json"), OUT.with_suffix(".md"), OUT_POST.with_suffix(".json"), OUT_POST.with_suffix(".md")]
    if not a.overwrite and any(p.exists() for p in outs):
        raise SystemExit("output exists; use --overwrite")
    o, post = build(a.codex_dir)
    # render from the stored form (sorted keys), as scripts/verify_report.py does
    o = json.loads(json.dumps(o, sort_keys=True))
    post = None if post is None else json.loads(json.dumps(post, sort_keys=True))
    OUT.with_suffix(".json").write_text(json.dumps(o, indent=1, sort_keys=True) + "\n")
    OUT.with_suffix(".md").write_text(render(o))
    if post is not None:
        OUT_POST.with_suffix(".json").write_text(json.dumps(post, indent=1, sort_keys=True) + "\n")
        OUT_POST.with_suffix(".md").write_text(render_post(post))
    print(json.dumps(o["chain_verdicts"], indent=1))
    print("relabelled types:", o["section1"]["relabelled_types"], "| post-audit:", o["post_audit_triggered"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
