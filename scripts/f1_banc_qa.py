"""F1 BANC QA report: every number of docs/results/f1_banc_qa.md is computed here (CLAUDE.md rule 11).

    env -u PYTHONPATH python scripts/f1_banc_qa.py [--out docs/results/f1_banc_qa]

Inputs: banc/neurons-banc.csv.gz, banc/connections_princeton-banc.csv.gz (hashes: configs/data_sources.yaml) and,
for the FlyWire comparison, brain_model/. Writes <out>.json and <out>.md (never overwrites without --overwrite).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from connectome import qa, qa_report as Q  # noqa: E402
from connectome.adapters import banc, flywire783 as fw  # noqa: E402

NEURONS = ROOT / "banc" / "neurons-banc.csv.gz"
CONNECTIONS = ROOT / "banc" / "connections_princeton-banc.csv.gz"
RULES = {"reference_shiu": (), "hist_inhibitory": ("hist_inhibitory",), "nt_impute": ("nt_impute",),
         "hist_inhibitory+nt_impute": ("hist_inhibitory", "nt_impute")}

# critical types: regex on the primary (or alternative) cell type, optionally restricted by annotation columns
WING_MN_PATTERNS = {"DLM": r"^DLM", "DVM": r"^DVM", "b1": r"^b1$", "b2": r"^b2$", "b3": r"^b3$", "i1": r"^i1$",
                    "i2": r"^i2$", "iii1": r"^iii1$", "iii2": r"^iii2$", "iii3": r"^iii3$", "iii4": r"^iii4$",
                    "hg1": r"^hg1$", "hg2": r"^hg2$", "hg3": r"^hg3$", "hg4": r"^hg4$", "tp1": r"^tp1$",
                    "tp2": r"^tp2$", "tpn": r"^tpn$", "ps1": r"^ps1$", "ps2": r"^ps2$", "PSI": r"^PSI$",
                    "PSn": r"^PSn"}


def critical_types(cx):
    n = cx.neurons
    motor = (n["super_class"] == "motor").to_numpy()
    bp = n["body_part"].astype(str)
    out = {}
    out["MN9"] = Q.count_types(cx, {"MN9": r"^MN9$"})["MN9"]
    wing = Q.count_types(cx, WING_MN_PATTERNS, use_alt=False, where=motor & (bp == "wing").to_numpy())
    out["wing_MN"] = wing
    out["wing_MN_all"] = dict(n=int((motor & (bp == "wing").to_numpy()).sum()))
    out["haltere_MN"] = dict(n=int((motor & (bp == "haltere").to_numpy()).sum()),
                             types=n[motor & (bp == "haltere").to_numpy()]["cell_type"].astype(str).value_counts().to_dict())
    out["neck_MN"] = dict(n=int((motor & (bp == "neck").to_numpy()).sum()))
    out["TTMn"] = Q.count_types(cx, {"TTMn": r"^TTMn$"})["TTMn"]
    out["GF (DNp01)"] = Q.count_types(cx, {"DNp01": r"^DNp01$"})["DNp01"]
    out["GF alt name 'giant_fiber'"] = Q.count_types(cx, {"gf": r"giant_fiber"}, use_alt=True)["gf"]
    out["DNg02"] = Q.count_types(cx, {"DNg02": r"^DNg02(?:_|$)"}, use_alt=False)["DNg02"]
    out["DNa02"] = Q.count_types(cx, {"DNa02": r"^DNa02$"})["DNa02"]
    out["DNp15"] = Q.count_types(cx, {"DNp15": r"^DNp15$"})["DNp15"]
    fn = n["function"].astype(str)
    sugar = fn.str.contains("sugar").to_numpy() & (n["super_class"].isin(["sensory", "sensory_ascending"])).to_numpy()
    lab = sugar & (bp == "labellum").to_numpy()
    leg = sugar & bp.isin(["front_leg", "middle_leg", "hind_leg"]).to_numpy()
    out["sugar_GRN_labellar"] = dict(n=int(lab.sum()), left=int((n["side"][lab] == "left").sum()),
                                     right=int((n["side"][lab] == "right").sum()))
    out["sugar_GRN_leg"] = dict(n=int(leg.sum()), left=int((n["side"][leg] == "left").sum()),
                                right=int((n["side"][leg] == "right").sum()),
                                by_leg=n[leg]["body_part"].value_counts().to_dict())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "results" / "f1_banc_qa"))
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    for suf in (".json", ".md"):
        if Path(str(out) + suf).exists() and not a.overwrite:
            raise SystemExit(f"{out}{suf} exists; use --overwrite or a new --out")
    R = {}
    print("BANC ...", flush=True)
    cx = banc.load_connectome(NEURONS, CONNECTIONS, hash_files=True)
    R["inputs"] = dict(sha256=cx.meta["sha256"])
    R["counts"] = Q.counts(cx)
    conn = banc.load_connections(CONNECTIONS)
    R["counts"]["neuropil_rows"] = int(len(conn))
    R["counts"]["neuropils"] = int(conn["neuropil"].nunique())
    R["counts"]["neuropil_rows_per_neuron"] = float(len(conn) / R["counts"]["neurons"])
    R["counts"]["region_of_neurons"] = cx.neurons["region"].value_counts(dropna=False).astype(str).to_dict()
    del conn

    # sign rules
    R["sign_rules"] = {}
    for name, var in RULES.items():
        c = banc.load_connectome(NEURONS, CONNECTIONS, variants=var, hash_files=False)
        R["sign_rules"][name] = dict(edge_signs=Q.sign_effects(c), nt_counts=c.meta["nt_counts"],
                                     applied=[{k: v for k, v in r.items() if k in ("name", "n_neurons", "n_edges_changed",
                                                                                 "n_neurons_changed", "syn_exc_to_inh",
                                                                                 "syn_inh_to_exc", "n_imputable")}
                                              for r in c.meta.get("sign_rules", [])])
        if name == "reference_shiu":
            ref = c
    cx = ref
    raw = pd.read_csv(NEURONS, low_memory=False, usecols=["Predicted NT type", "Verified NT type"])
    R["missing_nt"] = dict(
        missing=int(raw["Predicted NT type"].isna().sum()),
        with_verified_nt=int((raw["Predicted NT type"].isna() & raw["Verified NT type"].notna()).sum()))

    # symmetry
    t = Q.type_symmetry(cx)
    R["symmetry"] = Q.symmetry_summary(t)
    R["symmetry"]["types_without_input_on_one_side_all"] = int((~np.isfinite(t["log2_total"])).sum())
    R["symmetry"]["side_totals"] = Q.side_totals(cx)
    n_ = cx.neurons
    optic = n_["super_class"].isin(["optic_lobe_intrinsic", "visual_projection", "visual_centrifugal"])
    R["symmetry"]["optic_lobe_aggregate"] = dict(
        neurons_left=int((optic & (n_["side"] == "left")).sum()), neurons_right=int((optic & (n_["side"] == "right")).sum()))
    crit_t = t[t["type"].isin(["DNp15", "DNa02", "DNa01", "DNp01", "DNp07", "DNp10", "DNp03", "MN9", "TTMn", "H2",
                               "HSN", "HSE", "HSS"])]
    R["symmetry"]["critical_types"] = crit_t.drop(columns=["super_class"]).to_dict("records")
    R["symmetry"]["types_without_input_on_one_side"] = int((~np.isfinite(t["log2_total"])).sum())
    t = t[np.isfinite(t["log2_total"])].reset_index(drop=True)
    big = t[(t["n_left"] >= 10) & (t["n_right"] >= 10)].sort_values("log2_total")
    R["symmetry"]["largest_asymmetric_types_ge10_per_side"] = big.reindex(
        big["log2_total"].abs().sort_values(ascending=False).index).head(10)[
        ["type", "n_left", "n_right", "log2_total", "log2_exc"]].to_dict("records")

    # sparseness vs FlyWire
    print("FlyWire ...", flush=True)
    fx = fw.load_connectome(hash_files=False)
    bsc = n_["super_class"].replace({"optic_lobe_intrinsic": "optic", "central_brain_intrinsic": "central"})
    brain_like = n_["region"].isin(["brain", "SEZ"]).to_numpy()
    R["sparseness"] = dict(
        banc_by_region=Q.sparseness(cx, n_["region"].where(n_["region"].notna(), "none").to_numpy(object)),
        banc_by_region_ge5=Q.sparseness(cx, n_["region"].where(n_["region"].notna(), "none").to_numpy(object),
                                        min_syn=5),
        flywire_by_super_class=Q.sparseness(fx, fx.neurons["super_class"].to_numpy(object)),
        flywire_by_super_class_ge5=Q.sparseness(fx, fx.neurons["super_class"].to_numpy(object), min_syn=5),
        banc_by_super_class_brain=Q.sparseness(cx, np.where(brain_like, bsc.to_numpy(object), None)),
        banc_by_super_class_brain_ge5=Q.sparseness(cx, np.where(brain_like, bsc.to_numpy(object), None), min_syn=5),
        note="FlyWire v783 graph = Shiu model (threshold >= 5 synapses per pair); compare the *_ge5 blocks")

    # critical types
    R["critical_types"] = critical_types(cx)

    # antennal lobe structure (docs/02 section 1.6), same function on both datasets
    cls = n_["cell_class"].astype(str)
    R["antennal_lobe"] = dict(
        banc=Q.al_structure(cx, (cls == "antennal_lobe_local_neuron").to_numpy(),
                            (cls == "antennal_lobe_projection_neuron").to_numpy()),
        flywire=Q.al_structure(fx, (fx.neurons["cell_class"] == "ALLN").to_numpy(),
                               (fx.neurons["cell_class"] == "ALPN").to_numpy()))

    # docs/03 appendix A comparison (claims typed there; computed values from this run)
    sb, sf = R["sparseness"]["banc_by_region"]["__all__"], R["sparseness"]["flywire_by_super_class"]["__all__"]
    sb5, sf5 = R["sparseness"]["banc_by_region_ge5"]["__all__"], R["sparseness"]["flywire_by_super_class_ge5"]["__all__"]
    R["sparseness"]["flywire_over_banc"] = dict(
        in_edges_all_pairs=sf["in_edges_mean"] / sb["in_edges_mean"], in_syn_all_pairs=sf["in_syn_mean"] / sb["in_syn_mean"],
        in_edges_ge5=sf5["in_edges_mean"] / sb5["in_edges_mean"], in_syn_ge5=sf5["in_syn_mean"] / sb5["in_syn_mean"])
    A = dict(neurons=158_262, left=75_760, right=82_502, pairs=3_037_361, synapses=23_556_214,
             DN=1_316, motor=805, wing_MN=62, haltere_MN=25, neck_MN=49, TTMn=2, MN9=2, mean_in_edges=25,
             mean_in_syn=149, DNg02_family=38)
    got = dict(neurons=R["counts"]["neurons"], left=R["counts"]["left"], right=R["counts"]["right"],
               pairs=R["counts"]["pairs"], synapses=R["counts"]["synapses"],
               DN=R["counts"]["super_classes"].get("descending"), motor=R["counts"]["super_classes"].get("motor"),
               wing_MN=R["critical_types"]["wing_MN_all"]["n"], haltere_MN=R["critical_types"]["haltere_MN"]["n"],
               neck_MN=R["critical_types"]["neck_MN"]["n"], TTMn=R["critical_types"]["TTMn"]["n"],
               MN9=R["critical_types"]["MN9"]["n"],
               DNg02_family=R["critical_types"]["DNg02"]["n"],
               mean_in_edges=round(R["sparseness"]["banc_by_region"]["__all__"]["in_edges_mean"]),
               mean_in_syn=round(R["sparseness"]["banc_by_region"]["__all__"]["in_syn_mean"]))
    R["appendix_A_check"] = {k: dict(appendix_A=A[k], computed=got[k], equal=bool(A[k] == got[k])) for k in A}
    R["appendix_A_check"]["mean_in_edges"]["note"] = (
        f"appendix A divided neuropil rows by neurons ({R['counts']['neuropil_rows_per_neuron']:.1f}); unique pairs give "
        f"{R['sparseness']['banc_by_region']['__all__']['in_edges_mean']:.1f}")
    Path(str(out) + ".json").write_text(json.dumps(R, indent=1, default=str))
    Path(str(out) + ".md").write_text(render(R))
    print("written", out)


def render(R) -> str:
    c, s = R["counts"], R["symmetry"]
    L = ["# F1 BANC QA report", "",
         "Generated by `scripts/f1_banc_qa.py`; every number below is computed from the files whose SHA-256 are in "
         "`configs/data_sources.yaml`. Raw numbers: `f1_banc_qa.json`.", "",
         f"Inputs: neurons `{R['inputs']['sha256']['neurons'][:16]}...`, connections "
         f"`{R['inputs']['sha256']['connections'][:16]}...`.", "", "## 1. Counts",
         f"- neurons {c['neurons']:,} (left {c['left']:,}, right {c['right']:,}); typed {c['typed']:.1%}",
         f"- neuron pairs {c['pairs']:,}; synapses {c['synapses']:,}; median synapses per pair "
         f"{c['median_syn_per_pair']:.0f}; neuropil rows {c['neuropil_rows']:,} in {c['neuropils']} neuropils",
         f"- regions of neurons: {c['region_of_neurons']}", f"- super classes: {c['super_classes']}", "",
         "### Sign rules (edge signs; neurons = presynaptic neurons, synapses = outgoing)",
         "| rule | excitatory neurons / synapses | inhibitory neurons / synapses | applied |", "|---|---|---|---|"]
    for name, r in R["sign_rules"].items():
        e = r["edge_signs"]
        L.append(f"| {name} | {e['excitatory']['neurons']:,} / {e['excitatory']['synapses']:,} | "
                 f"{e['inhibitory']['neurons']:,} / {e['inhibitory']['synapses']:,} | {r['applied']} |")
    mv = R["missing_nt"]
    L += ["", f"Neurons without a predicted transmitter: {mv['missing']:,}, of which {mv['with_verified_nt']:,} have a "
          "`Verified NT type` in the export (not used: the registered rule reads the predicted type only).", "",
          "Neurons and outgoing synapses per transmitter label (reference rule):", "",
          "| transmitter | neurons | outgoing synapses |", "|---|---|---|"]
    for k, v in sorted(R["sign_rules"]["reference_shiu"]["nt_counts"].items(), key=lambda kv: -kv[1]["neurons"]):
        L.append(f"| {k} | {v['neurons']:,} | {v['out_synapses']:,} |")
    L += ["", "## 2. Left / right symmetry (input-synapse gate |log2(R/L)| < 0.5, total and excitatory)", "",
          "| set | types | pass | median abs log2 | > 0.5 | > 1 |", "|---|---|---|---|---|---|"]
    for lab in ("all", "n_ge10_per_side", "single_pair"):
        b = s[lab]
        L.append(f"| {lab} | {b['n_types']} | {b['pass_fraction']:.2f} | {b['median_abs_log2_total']:.2f} | "
                 f"{b['frac_abs_log2_gt_0p5']:.2f} | {b['frac_abs_log2_gt_1']:.2f} |")
    for sc, b in s["by_super_class"].items():
        L.append(f"| super class {sc} | {b['n_types']} | {b['pass_fraction']:.2f} | {b['median_abs_log2_total']:.2f} | "
                 f"{b['frac_abs_log2_gt_0p5']:.2f} | {b['frac_abs_log2_gt_1']:.2f} |")
    L += ["", "Input synapses and neuron counts per super class and side (optic lobe included):", "",
          "| super class | neurons L / R | input synapses R/L |", "|---|---|---|"]
    for sc, v in s["side_totals"].items():
        L.append(f"| {sc} | {v['n_left']:,} / {v['n_right']:,} | {v['ratio_in_right_over_left']:.2f} |")
    L += ["", f"Types with neurons on both sides but no input synapses on one side (excluded from the log2 statistics "
          f"of the lists below): {s['types_without_input_on_one_side']}.", "",
          "Critical types (types with neurons on both sides):", "",
          "| type | n L / R | log2 total | log2 exc | gate |", "|---|---|---|---|---|"]
    for r in s["critical_types"]:
        L.append(f"| {r['type']} | {r['n_left']} / {r['n_right']} | {r['log2_total']:+.2f} | {r['log2_exc']:+.2f} | "
                 f"{'pass' if r['pass'] else 'FAIL'} |")
    L += ["", "Most asymmetric types with >= 10 neurons per side:", "", "| type | n L / R | log2 total | log2 exc |",
          "|---|---|---|---|"]
    for r in s["largest_asymmetric_types_ge10_per_side"]:
        L.append(f"| {r['type']} | {r['n_left']} / {r['n_right']} | {r['log2_total']:+.2f} | {r['log2_exc']:+.2f} |")
    sp = R["sparseness"]
    fb = sp["flywire_over_banc"]
    L += ["", "## 3. Sparseness (mean per neuron)", "", f"{sp['note']}", "",
          f"FlyWire / BANC, all neurons: in-edges {fb['in_edges_all_pairs']:.2f}x and in-synapses "
          f"{fb['in_syn_all_pairs']:.2f}x with all pairs; in-edges {fb['in_edges_ge5']:.2f}x and in-synapses "
          f"{fb['in_syn_ge5']:.2f}x at >= 5 synapses per pair.", "",
          "| dataset / group | neurons | in-edges | in-synapses | synapses per edge |", "|---|---|---|---|---|"]
    for blk in ("banc_by_region", "banc_by_region_ge5", "banc_by_super_class_brain", "banc_by_super_class_brain_ge5",
                "flywire_by_super_class", "flywire_by_super_class_ge5"):
        for k, v in sp[blk].items():
            if k in ("None", "none") or v["n"] < 100:
                continue
            L.append(f"| {blk} / {k} | {v['n']:,} | {v['in_edges_mean']:.1f} | {v['in_syn_mean']:.1f} | "
                     f"{v['syn_per_edge']:.2f} |")
    ct = R["critical_types"]
    L += ["", "## 4. Critical types", "", "| item | n | left | right | note |", "|---|---|---|---|---|"]
    for k in ("MN9", "TTMn", "GF (DNp01)", "GF alt name 'giant_fiber'", "DNg02", "DNa02", "DNp15",
              "sugar_GRN_labellar", "sugar_GRN_leg"):
        v = ct[k]
        L.append(f"| {k} | {v['n']} | {v.get('left', '')} | {v.get('right', '')} | {v.get('by_leg', v.get('types', ''))} |")
    L.append(f"| haltere MN (motor, body part haltere) | {ct['haltere_MN']['n']} | | | {ct['haltere_MN']['types']} |")
    L.append(f"| neck MN (motor, body part neck) | {ct['neck_MN']['n']} | | | |")
    L.append(f"| wing MN (motor, body part wing), all | {ct['wing_MN_all']['n']} | | | |")
    for k, v in ct["wing_MN"].items():
        L.append(f"| wing MN {k} | {v['n']} | {v['left']} | {v['right']} | {v['types']} |")
    al = R["antennal_lobe"]
    L += ["", "## 5. Antennal-lobe structure (docs/02 section 1.6 check, reference sign rule)", "",
          "| quantity | BANC | FlyWire v783 |", "|---|---|---|"]
    rows = [("ALLN / ALPN", lambda d: f"{d['n_alln']} / {d['n_alpn']}"),
            ("ALLN sign exc / inh", lambda d: f"{d['alln_sign']['excitatory']} / {d['alln_sign']['inhibitory']}"),
            ("ALPN sign exc / inh", lambda d: f"{d['alpn_sign']['excitatory']} / {d['alpn_sign']['inhibitory']}"),
            ("internal LN+PN synapses exc / inh",
             lambda d: f"{d['internal_syn']['excitatory']:,.0f} / {d['internal_syn']['inhibitory']:,.0f}"),
            ("excitatory fraction", lambda d: f"{d['internal_syn']['excitatory_fraction']:.1%}"),
            ("neurons with net recurrent drive >= threshold at 100 Hz",
             lambda d: f"{d['neurons_with_net_rec_ge_threshold_at_100Hz']} of {d['of']}"),
            ("leading eigenvalue, signed (synapses)", lambda d: f"{d['leading_eig_signed_syn']:,.0f}"),
            ("leading eigenvalue, excitatory only (synapses)", lambda d: f"{d['leading_eig_exc_syn']:,.0f}"),
            ("self-sustaining mean rate (Hz)",
             lambda d: "-" if d["self_sustaining_mean_rate_hz"] is None else f"{d['self_sustaining_mean_rate_hz']:.2f}")]
    for lab, f in rows:
        L.append(f"| {lab} | {f(al['banc'])} | {f(al['flywire'])} |")
    L += ["", "## 6. Comparison with docs/03 appendix A", "", "| quantity | appendix A | computed | equal | note |",
          "|---|---|---|---|---|"]
    for k, v in R["appendix_A_check"].items():
        L.append(f"| {k} | {v['appendix_A']:,} | {v['computed']:,} | {'yes' if v['equal'] else 'NO'} | "
                 f"{v.get('note', '')} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
