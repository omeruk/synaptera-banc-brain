"""Data-only QA for prereg/05: the v3 full graph (``banc888_full_v3``) and the per-neuron input normalisation factors
(no simulation).

    env -u PYTHONPATH python scripts/f2_v3_qa.py --out docs/results/f2_v3_qa

Primary neuron set (158,262 Codex neurons, edges among them, no threshold). Graph size and FlyWire / BANC in-synapse
ratio per super class for the Codex export, the v2 full graph and the v3 full graph; the factors ``s`` of arm N on the
v2 and v3 graphs (distribution per class, clipped neurons, ratio after scaling) and the antennal-lobe leading
eigenvalue after scaling next to FlyWire's. Markdown and JSON are written by this script.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from connectome import innorm, qa_report as Q  # noqa: E402
from connectome.adapters import banc, flywire783 as fw  # noqa: E402
from f2_wsyn_k import BANC_SC, CLASSES, in_synapses, ratio_table  # noqa: E402

B = ROOT / "banc"
NEURONS, CONN, META = B / "neurons-banc.csv.gz", B / "connections_princeton-banc.csv.gz", B / "banc_888_meta.feather"
V2, V3 = B / "banc_888_edgelist_simple_v2.feather", B / "banc_888_edgelist_simple_v3.feather"


def view(cx):
    return dict(neurons=int(cx.n_neurons), **Q.pair_threshold_profile(cx, (1, 3, 5)),
                median_syn_per_pair=float(cx.edges["n_syn"].median()),
                mean_in_edges=float(np.bincount(cx.edge_endpoints()[1], minlength=cx.n_neurons).mean()),
                mean_in_syn=float(in_synapses(cx).mean()))


def al(cx):
    cls = cx.neurons["cell_class"].astype(str)
    if cx.name == "flywire783":
        a, p = (cls == "ALLN"), (cls == "ALPN")
    else:
        a, p = (cls == "antennal_lobe_local_neuron"), (cls == "antennal_lobe_projection_neuron")
    return a.to_numpy(), p.to_numpy()


def al_eig(cx, s=None):
    v = cx if s is None else innorm.scaled_view(cx, s)
    if s is not None:
        v.edges["sign"] = cx.edges["sign"].to_numpy()
    a, p = al(cx)
    r = Q.al_structure(v, a, p)
    return dict(leading_eig_signed_syn=r["leading_eig_signed_syn"], leading_eig_exc_syn=r["leading_eig_exc_syn"],
                n_alln=r["n_alln"], n_alpn=r["n_alpn"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if (out.with_suffix(".md").exists() or out.with_suffix(".json").exists()) and not a.overwrite:
        raise SystemExit(f"{out}.* exists")
    fx = fw.load_connectome(hash_files=False)
    fsc = fx.neurons["super_class"].astype(object).to_numpy()
    ftype = fx.neurons["cell_type"].astype(object).to_numpy()
    R = dict(flywire=view(fx), banc={}, innorm={}, antennal_lobe=dict(flywire=al_eig(fx)))
    graphs = {"codex_export_ge3": banc.load_connectome(NEURONS, CONN, hash_files=False),
              "banc888_full_v2": banc.load_connectome_full(NEURONS, V2, META, CONN, "codex", hash_files=False),
              "banc888_full_v3": banc.load_connectome_full(NEURONS, V3, META, CONN, "codex", hash_files=False,
                                                           data_source="banc888_full_v3")}
    for name, cx in graphs.items():
        bsc = cx.neurons["super_class"].replace(BANC_SC).astype(object).to_numpy()
        btype = cx.neurons["cell_type"].astype(object).to_numpy()
        R["banc"][name] = dict(**view(cx), k=ratio_table(fx, cx, fsc, bsc, ftype, btype, 0))
        R["antennal_lobe"][name] = dict(unscaled=al_eig(cx))
        if name.startswith("banc888"):
            s, rep = innorm.innorm_factors(fx, cx)
            R["innorm"][name] = rep
            R["antennal_lobe"][name]["innorm"] = al_eig(cx, s)
            # in-synapse ratio after scaling (matched types): FlyWire / (BANC in-synapses x s[post])
            scaled = in_synapses(cx) * s
            R["banc"][name]["k_after_innorm"] = ratio_table(fx, cx, fsc, bsc, ftype, btype, 0, bin_override=scaled)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(R, indent=1))
    f = R["flywire"]
    L = ["# F2 prep: v3 full graph and input normalisation, data-only QA (`scripts/f2_v3_qa.py`)", "",
         "Primary neuron set: the 158,262 Codex neurons, edges among them, no threshold. `banc888_full_v3` = "
         "`banc_888_edgelist_simple_v3.feather` (v3 synapse set, `size >= 10`; Harvard Dataverse "
         "doi:10.7910/DVN/7WTH1N). Numbers are written by the script; criteria are those of "
         "`prereg/05_banc_v3_and_innorm.md`.", "",
         "## 1. Graph size", "",
         "| graph | neurons | pairs | synapses | median syn/pair | pairs >= 3 | pairs >= 5 | mean in-edges | mean in-syn |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, v in [("FlyWire v783 (Shiu)", f)] + list(R["banc"].items()):
        L.append(f"| {name} | {v['neurons']:,} | {v['pairs']:,} | {v['synapses']:,} | {v['median_syn_per_pair']:.0f} | "
                 f"{v['ge3']['pairs']:,} | {v['ge5']['pairs']:,} | {v['mean_in_edges']:.1f} | {v['mean_in_syn']:.1f} |")
    for view_name, label in (("matched", "matched cell types"), ("all", "all neurons of the class")):
        cols = list(R["banc"])
        L += ["", f"## 2{'a' if view_name == 'matched' else 'b'}. FlyWire / BANC ratio of mean input synapses per "
              f"postsynaptic neuron, {label}", "",
              "| class | " + " | ".join(cols) + " | v2 after N | v3 after N |", "|---|" + "---|" * (len(cols) + 2)]
        for c in ("__all__",) + CLASSES:
            cells = []
            for k in cols:
                r = R["banc"][k]["k"][view_name][c]
                cells.append("n/a" if r["k"] is None else f"{r['k']:.2f} ({r['n_banc']:,})")
            for k in ("banc888_full_v2", "banc888_full_v3"):
                r = R["banc"][k]["k_after_innorm"][view_name][c]
                cells.append("n/a" if r["k"] is None else f"{r['k']:.2f}")
            L.append(f"| {'all' if c == '__all__' else c} | " + " | ".join(cells) + " |")
    L += ["", "Values are FlyWire / BANC (number of BANC neurons). Ascending and motor include VNC compartments that "
          "FlyWire (brain only) lacks. The two last columns multiply the BANC input synapses of each neuron by its "
          "factor `s` before the ratio (arm N); the clip makes classes with a ratio outside [0.25, 4] stay off 1.", ""]
    L += ["## 3. Arm N factors `s` (`connectome/innorm.py`, computed from data only)", ""]
    for name, rep in R["innorm"].items():
        L += [f"### {name}", "",
              f"{rep['n_homolog']:,} of {rep['n_neurons']:,} neurons have a homolog type ({rep['n_homolog_types']:,} types); "
              f"{rep['n_fallback']:,} use the super-class median. Clipped: {rep['n_clipped_low']:,} low (0.25), "
              f"{rep['n_clipped_high']:,} high (4), {100 * rep['frac_clipped']:.1f} % of neurons. Global median of homolog "
              f"s: {rep['global_median_s']:.2f}. s of MN9: "
              + (f"{rep['named']['CB0701']:.2f}" if rep["named"]["CB0701"] else "n/a")
              + ", LB3: " + (f"{rep['named']['LB3']:.2f}" if rep["named"]["LB3"] else "n/a")
              + f". SHA-256 of s: `{rep['s_sha256'][:16]}…`.", "",
              "| super class | n | mean s | q05 | q25 | median | q75 | q95 | homolog n | fallback n | clipped low / high |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for c, d in sorted(rep["by_super_class"].items(), key=lambda kv: -kv[1]["all"]["n"]):
            a_ = d["all"]
            L.append(f"| {c} | {a_['n']:,} | {a_['mean']:.2f} | {a_['q05']:.2f} | {a_['q25']:.2f} | {a_['q50']:.2f} | "
                     f"{a_['q75']:.2f} | {a_['q95']:.2f} | {d['homolog']['n']:,} | {d['fallback']['n']:,} | "
                     f"{d['n_clipped_low']:,} / {d['n_clipped_high']:,} |")
        L.append("")
    L += ["## 4. Antennal-lobe leading eigenvalue (ALLN + ALPN subnetwork, reference sign rule; synapse units)", "",
          "| graph | signed | excitatory only | ALLN / ALPN |", "|---|---|---|---|"]
    fa = R["antennal_lobe"]["flywire"]
    L.append(f"| FlyWire v783 | {fa['leading_eig_signed_syn']:,.0f} | {fa['leading_eig_exc_syn']:,.0f} | "
             f"{fa['n_alln']} / {fa['n_alpn']} |")
    for name, d in R["antennal_lobe"].items():
        if name == "flywire":
            continue
        u = d["unscaled"]
        L.append(f"| {name} | {u['leading_eig_signed_syn']:,.0f} | {u['leading_eig_exc_syn']:,.0f} | "
                 f"{u['n_alln']} / {u['n_alpn']} |")
        if "innorm" in d:
            n_ = d["innorm"]
            L.append(f"| {name} after N | {n_['leading_eig_signed_syn']:,.0f} | {n_['leading_eig_exc_syn']:,.0f} | "
                     f"{n_['n_alln']} / {n_['n_alpn']} |")
    L.append("")
    out.with_suffix(".md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    raise SystemExit(main())
