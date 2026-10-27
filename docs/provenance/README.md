# Provenance of the carried Synaptera v1 code

This repository continues Synaptera v1.0.0. Until this PR it held a reference copy of the v1 files under `vendor/synaptera_v1/`. The copy was never imported, and it was removed in this PR.

The removed files are unchanged in two places:

- the v1 repository at commit `c94796d`: https://github.com/omeruk/synaptera/tree/c94796d480ad2e77476bb110ab9a1bbf52e27118;
- this repository's history, from commit `f222589` (the initial skeleton) on.

The three files in this folder are the provenance record, moved from `vendor/synaptera_v1/` and given the prefix `synaptera_v1_`:

- `synaptera_v1_MANIFEST.md`: the source commit and the list of carried files.
- `synaptera_v1_NOTICE.md`: the licences and attributions of the third-party data, models and software of v1.
- `synaptera_v1_THIRD_PARTY.md`: the full audit of those components, with their licences and provenance.

They are kept unedited, so their internal links follow the layout of the v1 repository, not this one.
