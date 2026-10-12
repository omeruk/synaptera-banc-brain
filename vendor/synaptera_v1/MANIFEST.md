# vendor/synaptera_v1 — Synaptera v1.0.0'dan değiştirilmeden kopyalanan dosyalar

Kaynak: https://github.com/omeruk/synaptera, commit `c94796d480ad2e77476bb110ab9a1bbf52e27118`. Bu dosyalar **referanstır**, içe aktarılmaz;
yeni kod proje düzeninde yazılır ve her dosyanın başına kaynağı (yol + bu commit) not edilir.

| dosya | karar | hedef | SHA-256 |
|---|---|---|---|
| `flight/groups.py` | adapt | `connectome/schema.py, connectome/homologs.py` | `7da5835867f66351…` |
| `scripts/make_descending_neurons.py` | adapt | `connectome/adapters/flywire783.py` | `8ab1f2dd5240af96…` |
| `scripts/make_sugar_grn.py` | adapt | `scripts/make_crosswalk.py` | `8c16cb504c77ef40…` |
| `scripts/make_nt_silent.py` | pattern | `connectome/sign_rules.py` | `a6cdd62e3870f651…` |
| `scripts/make_nt_impute.py` | pattern | `connectome/sign_rules.py` | `3e01d03c295d2f07…` |
| `scripts/make_nt_literature.py` | pattern | `connectome/sign_rules.py` | `ceabea735cc80aad…` |
| `flight/brain.py` | adapt | `engine/lif_brian2.py, engine/protocol.py` | `8ab3ac0ad4c74245…` |
| `flight/config.py` | adapt | `configs/arena_x20.yaml, body/arena.py` | `12cd85b86d4e702f…` |
| `flight/quasi_steady.py` | adapt | `body/quasi_steady.py, motor/legacy_reflexes.py` | `1b3be7611185fdef…` |
| `flight/body.py` | adapt | `body/flygym_body.py, body/arena.py` | `d8b0ce5d9c92765d…` |
| `flight/head_reflex.py` | adapt | `motor/legacy_reflexes.py` | `c44c3ba5c6a5c327…` |
| `flight/visual_input.py` | adapt | `sense/vision_flyvis.py` | `edf5c26fc00af0a8…` |
| `flight/vision_boundary.py` | adapt | `sense/vision_boundary.py` | `1ebac5de97f9dc8f…` |
| `scripts/make_t45_transduction.py` | adapt | `scripts/make_transduction.py` | `8766927fc5bd2191…` |
| `scripts/make_visual_transduction.py` | adapt | `scripts/make_transduction.py` | `e5ed229fcafb781a…` |
| `scripts/make_vision_boundary.py` | adapt | `scripts/make_vision_boundary.py` | `7111f8488b0ef430…` |
| `scripts/prepare_neuron_geometry.py` | pattern | `connectome/geometry.py` | `e3439dbadd0acfa8…` |
| `flight/vis_stim.py` | take | `tasks/t0_stimuli.py` | `4bf718cfa256ba77…` |
| `flight/sensors.py` | adapt | `sense/ (gaze, odor örnekleme, yüzey, genişleme)` | `401b6b058791f27e…` |
| `simulation_data/odor_field_3d.py` | adapt | `sense/odor_field.py` | `5fbfb25e148c259a…` |
| `flight/olfaction_full.py` | adapt | `sense/olfaction.py` | `48d43948035b38f4…` |
| `scripts/make_orn_spontaneous.py` | adapt | `scripts/make_orn_spontaneous.py` | `a36306c2ba960a45…` |
| `flight/readouts.py` | adapt | `motor/readouts.py` | `e09320830835d7a2…` |
| `flight/vnc_bridge.py` | adapt | `motor/legacy_bridge.py` | `910620d81a27d4de…` |
| `fly_flight_brain_body_simulation.py` | pattern | `coupling/scheduler.py, run.py` | `e8be037cff0885f4…` |
| `flight/recorder.py` | adapt | `coupling/recorder.py` | `1572c25941953f0d…` |
| `scripts/ladder_replay.py` | pattern | `coupling/replay.py` | `55264f35ffab7621…` |
| `flight/readout_model.py` | adapt | `train/readout_ridge.py` | `2a83f08639facf60…` |
| `scripts/ladder_fit.py` | pattern | `train/` | `36e2a401403debd6…` |
| `scripts/make_dn_reference.py` | pattern | `tasks/t0_openloop.py` | `711cd01e13f08665…` |
| `scripts/fetch_data.py` | adapt | `scripts/fetch_data.py` | `7be3aaba503f4413…` |
| `scripts/verify_report_final.py` | pattern | `scripts/verify_report.py` | `f62bcafeb47ee989…` |
| `render_flight_video_v2.py` | pattern | `scripts/render_video.py` | `9757adb2f850aa5f…` |
| `requirements-flight.txt` | adapt | `requirements.txt` | `c4d50f6ba82fe9c7…` |
| `tests/conftest.py` | adapt | `tests/conftest.py` | `6aebbbdae83e730b…` |
| `tests/flight/test_body.py` | adapt | `tests/test_body.py` | `c274dd10f5f11233…` |
| `tests/flight/test_start_pose.py` | adapt | `tests/test_body_pose.py` | `575af70b1a6e436c…` |
| `tests/flight/test_hybrid.py` | adapt | `tests/test_hand_scaffold.py` | `a5a704a8bb4861c1…` |
| `tests/flight/test_brain.py` | adapt | `tests/test_engine_lif.py` | `1d00aa27e17e0d6c…` |
| `tests/flight/test_readouts.py` | adapt | `tests/test_readouts.py` | `96c237ed42224bfc…` |
| `tests/flight/test_vnc_bridge.py` | adapt | `tests/test_legacy_bridge.py` | `0cefc45ceb01e7de…` |
| `tests/flight/test_visual_input.py` | adapt | `tests/test_vision_flyvis.py` | `a555efe7c75e0a55…` |
| `tests/flight/test_vision_boundary.py` | adapt | `tests/test_vision_boundary.py` | `ecead0318e383368…` |
| `tests/flight/test_vis_stim.py` | take | `tests/test_t0_stimuli.py` | `76f304a3c18d6c01…` |
| `tests/flight/test_vis_stim_followup.py` | take | `tests/test_t0_stimuli_followup.py` | `e36d3dba406dab6d…` |
| `tests/flight/test_readout_model.py` | adapt | `tests/test_readout_ridge.py` | `c8d53770dc978c6c…` |
| `tests/flight/test_readout_record.py` | pattern | `tests/test_recorder.py` | `4fd0d39f455b1019…` |
| `tests/flight/test_smoke_hdf5.py` | pattern | `tests/test_smoke_run.py` | `b20178b054bed9aa…` |
| `tests/flight/test_odor_field_3d.py` | adapt | `tests/test_odor_field.py` | `d2e052b450a5f073…` |
| `tests/flight/test_olfaction_full.py` | adapt | `tests/test_olfaction.py` | `65e421edb54a5b6a…` |
| `THIRD_PARTY.md` | pattern | `THIRD_PARTY.md` | `02a931f0ba9f190c…` |
| `NOTICE.md` | pattern | `NOTICE.md` | `1cf373ec863c59d3…` |
