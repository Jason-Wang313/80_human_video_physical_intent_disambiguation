# Plan

Build paper 80 `human_video_physical_intent_disambiguation` from the shared pool, compile PDF to Downloads only, and publish the exact-name public repo.

## 2026-06-15 Continuation Plan and Result

Plan before execution: re-audit Paper 80 under the ICLR-main-target gate without reducing experiment quality. The required checks were code compilation, full deterministic experiment rerun, CSV row/schema/seed/method coverage, main baseline comparisons, paired uncertainty, morphology-leakage probes, ablations, stress sweeps, PDF/BibTeX cleanliness, Downloads-only artifact placement, Desktop exclusion, and public GitHub readiness.

Result: `python -m py_compile src/run_experiment.py` passed and `python src/run_experiment.py` regenerated 10,290 main rollout rows, 2,058 ablation rollout rows, and 20,160 stress rollout rows. The local evidence still supports STRONG_REVISE: `physical_intent_disambiguation` reaches 0.77891 +/- 0.02828 action success on `combined_hard_shift`, compared with 0.60544 +/- 0.04303 for `object_affordance_classifier` and 0.48639 +/- 0.05837 for `style_invariant_logistic`. The paired gain over object-affordance classification is 0.17347 +/- 0.02211 with 7/7 better seeds. The paper remains not ICLR-main-ready because no real human-video, robot, external learned-baseline, or recognized high-fidelity benchmark evidence is present.
