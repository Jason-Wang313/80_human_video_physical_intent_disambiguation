# Paper 80 Terminal Audit - 2026-06-15

## Scope

Paper 80, `human_video_physical_intent_disambiguation`, was re-audited under the sequential ICLR-main-target continuation gate. The audit tested whether the local keypoint-video evidence remains strong enough for `STRONG_REVISE`, and whether any actual evidence exists that could upgrade the paper to ICLR-main-ready.

## Plan

The execution plan was written first in `docs/paper80_iclr_submission_execution_plan_20260615.md`. The plan required code compilation, a full deterministic experiment rerun, CSV integrity checks, main baseline analysis, paired uncertainty, morphology-leakage probes, ablations, stress sweeps, PDF hygiene, Downloads-only artifact placement, Desktop exclusion, public GitHub verification, and report updates.

## Verification

- Code compilation: `python -m py_compile src/run_experiment.py` passed.
- Experiment rerun: `python src/run_experiment.py` completed and returned terminal recommendation STRONG_REVISE.
- Main rows: 10,290 rollout rows, 1,470 dataset-summary rows, 210 morphology-leakage rows, 245 seed-level metric rows, 205 aggregate-metric rows, and 75 pairwise-stat rows.
- Ablation rows: 2,058 ablation rollout rows and 7 ablation summary rows.
- Stress rows: 20,160 stress rollout rows and 120 stress summary rows.
- Negative cases: 16 rows.
- Seeds: 0 through 6.
- Methods: `body_normalized_knn`, `object_affordance_classifier`, `oracle_intent_upper_bound`, `physical_intent_disambiguation`, `raw_trajectory_knn`, `style_invariant_logistic`, and `velocity_contact_classifier`.

## Central Evidence

On `combined_hard_shift`, `physical_intent_disambiguation` reaches 0.77891 +/- 0.02828 action success. The strongest non-oracle baseline, `object_affordance_classifier`, reaches 0.60544 +/- 0.04303, and `style_invariant_logistic` reaches 0.48639 +/- 0.05837.

The paired action-success gain over `object_affordance_classifier` is 0.17347 +/- 0.02211 with 7/7 better seeds. The paired gain over `style_invariant_logistic` is 0.29252 +/- 0.06498 with 7/7 better seeds.

Morphology leakage drops from 0.78231 for raw trajectory features to 0.57823 for the factored method. The object-affordance ablation drops action success from 0.77891 to 0.66327, supporting the core object/factoring mechanism. Contact timing and uncertainty gating are neutral in this local benchmark and should not be overclaimed.

At maximum combined stress, the proposed method reaches 0.71429 +/- 0.04518 action success, compared with 0.51786 +/- 0.08139 for `object_affordance_classifier` and 0.47619 +/- 0.05872 for `style_invariant_logistic`.

## Artifact Verification

- Downloads PDF: `C:/Users/wangz/Downloads/80.pdf`
- SHA256: `C3BA4F78792B565CEC2DC80658E3AF551D0F792EABF66F0C0783039A2684B0DC`
- Desktop PDF: absent at `C:/Users/wangz/Desktop/80.pdf`
- GitHub: `https://github.com/Jason-Wang313/80_human_video_physical_intent_disambiguation`

## Decision

Final decision: STRONG_REVISE.

The local evidence is positive and reproducible, so the paper should not be archived. It is still not ICLR-main-ready because the evidence is generated local keypoint-video data, with no real human-video demonstrations, no robot hardware validation, no recognized high-fidelity benchmark, and no external learned video-to-robot baseline.
