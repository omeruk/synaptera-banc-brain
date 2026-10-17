# Errata 01 (post-hoc) — prereg/14, prereg/17 and docs/10

Status: **post-hoc** (CLAUDE.md rule 6), written 2026-10-17 after PR #24 was merged. No pre-registration file is
edited; this file records two factual errors in their text. Neither changes a criterion, a decision rule, a check
result or the design.

## E1. prereg/17 section 3: zip timestamps of `.npz` files

- **The text says** that the `.npz` file hashes are reported but are not the criterion, because
  "`np.savez_compressed` writes the write time into each zip member header".
- **This is wrong.** numpy 2.0.2 opens each member with `zipfile.ZipFile.open(name, "w")`. That creates a
  `zipfile.ZipInfo(name)` whose default `date_time` is the fixed date 1980-01-01 00:00:00. The write time is never
  stored, so two identical records give identical files.
- **Evidence:**
  - Both records of the prereg/17 section 4 check carry the member date `(1980, 1, 1, 0, 0, 0)`.
  - Their file SHA-256s are equal (`d58341a0b544643c...`), as PR #24 reports.
- **Effect.** None:
  - The registered criterion (identical members and identical uncompressed `.npy` bytes per member) stands, and the
    check passed under it.
  - The file hashes are equal as well. They stay descriptive, as registered.

## E2. prereg/14 and docs/10: the `ensemble` bridge has 225 units, not 224

- **The text says 224 units** in these places:
  - prereg/14 section 2: "the prereg/11 `ensemble` (112 families, 224 units, 305 neurons)";
  - prereg/14 section 9, gate 1: "Ridge on the window-mean 224 bridge units";
  - docs/10 section 2.3: "224 family-side units";
  - docs/10 section 5: "Stage A input: the 224 bridge units";
  - docs/10 section 9: "ridge regression on the 224 filtered bridge units".
- **The `ensemble` bridge has 225 units:** 221 matched by exact type and 4 by family, over 112 families and 305 BANC
  neurons (`docs/results/f3b2_dn_ensemble.md`, `docs/results/f3b3_specificity.md`).
- **224 is the unit count of other bridges:** the derangement bridges `random_1` ... `random_39` and
  `ensemble_shuffled`, which match by source family only. `docs/results/f5_reachability.md` already noted this.
- **What is binding.** prereg/14 defines the bridge as the prereg/11 `ensemble`. That definition holds, and the count
  was a description error. Wherever the texts above say "224 bridge units", read the units of the `ensemble` bridge
  (225), in particular:
  - the gate-1 ridge input;
  - the `mlp` and `linear` input in stage A (225 units x 4 filters).
- **The `derange` arm** uses the prereg/11 derangement bridges, so its unit count is 224, as built.
- **Unchanged:** the families, the neurons, the matching rule, the window and the latency.
