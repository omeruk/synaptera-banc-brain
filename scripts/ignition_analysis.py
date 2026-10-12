"""Analysis of the pre-registered ignition test (prereg/02_ignition_frequency.md) -> docs/results/f0_ignition_frequency.{json,md}."""
import json
from pathlib import Path

from scipy.stats import binomtest, fisher_exact

ROOT = Path(__file__).resolve().parent.parent
ARMS = ("v1", "new", "new_delay")


def main():
    data = {}
    for a in ARMS:
        rows = [json.loads(l) for l in (ROOT / "runs/ignition" / f"{a}.jsonl").read_text().splitlines() if l.strip()]
        rows = {r["seed"]: r for r in rows}
        assert sorted(rows) == list(range(100, 200)), f"{a}: seeds {len(rows)}"
        data[a] = rows
    res = {}
    for a, rows in data.items():
        k = sum(r["ignited_primary"] for r in rows.values())
        k2 = sum(r["ignited_secondary"] for r in rows.values())
        ci = binomtest(k, 100).proportion_ci(confidence_level=0.95, method="exact")
        res[a] = dict(ignited=k, n=100, proportion=k / 100, ci95=[ci.low, ci.high], secondary_ignited=k2,
                      ignited_seeds=sorted(s for s, r in rows.items() if r["ignited_primary"]))

    def fisher(x, y):
        t = [[res[x]["ignited"], 100 - res[x]["ignited"]], [res[y]["ignited"], 100 - res[y]["ignited"]]]
        return float(fisher_exact(t, alternative="two-sided")[1])
    tests = {"primary_v1_vs_new": fisher("v1", "new"), "new_delay_vs_v1": fisher("new_delay", "v1"),
             "new_delay_vs_new": fisher("new_delay", "new")}
    out = dict(arms=res, fisher_two_sided=tests, alpha=0.05,
               primary_significant=bool(tests["primary_v1_vs_new"] < 0.05))
    (ROOT / "docs/results/f0_ignition_frequency.json").write_text(json.dumps(out, indent=1))
    lines = ["| arm | ignited / 100 | proportion | 95 % CI (Clopper-Pearson) | secondary (idle mean > 0.5 Hz) | seeds |",
             "|---|---|---|---|---|---|"]
    for a, r in res.items():
        lines.append(f"| {a} | {r['ignited']} | {r['proportion']:.2f} | {r['ci95'][0]:.3f}-{r['ci95'][1]:.3f} | "
                     f"{r['secondary_ignited']} | {r['ignited_seeds']} |")
    lines += ["", f"Fisher exact, two-sided: v1 vs new (primary) p = {tests['primary_v1_vs_new']:.4f}; "
                  f"new_delay vs v1 p = {tests['new_delay_vs_v1']:.4f}; new_delay vs new p = {tests['new_delay_vs_new']:.4f}.",
              f"Primary test at alpha 0.05: {'SIGNIFICANT' if out['primary_significant'] else 'not significant'}."]
    (ROOT / "docs/results/f0_ignition_frequency.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
