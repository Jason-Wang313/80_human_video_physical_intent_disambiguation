# Paper 80 Rebuild Plan

## Goal

Rebuild `human_video_physical_intent_disambiguation` from an archive-only synthetic scaffold into the strongest honest ICLR-main-target artifact possible. The concrete scientific question is whether robot policies can infer task-level physical intent from human video while suppressing morphology-specific and style-specific motion artifacts.

## Evidence Standard

The rebuild must replace scalar probability diagnostics with an implemented benchmark, implemented baselines, multi-seed evaluation, ablations, stress tests, uncertainty summaries, and a reproducible paper PDF. Because this workspace does not contain real human-video data, hardware logs, or high-fidelity simulator assets, the ceiling is `STRONG_REVISE`; if the proposed mechanism fails to clear strong baselines or depends on synthetic shortcuts, the correct terminal decision is `KILL_ARCHIVE`.

## Benchmark

Create a deterministic local kinematic-video benchmark with generated human keypoint trajectories, object state, contact timing, and robot-action targets. Each episode will include morphology variables, style variables, camera/occlusion noise, object affordance parameters, and a ground-truth physical intent.

Evaluation splits:

1. `seen_morphology_clean`: familiar body scale, handedness, style, and object affordances.
2. `style_shift_same_intent`: speed, curvature, hesitation, and path-shape shifts without intent shift.
3. `morphology_shift`: arm length, shoulder width, height, handedness, and camera-scale shifts.
4. `object_affordance_ambiguity`: visually similar reaches where object constraints distinguish push, pull, rotate, lift, place, or handoff intent.
5. `combined_hard_shift`: morphology, style, affordance ambiguity, occlusion, noise, and partial observation combined.

## Methods

Implement all methods in `src/run_experiment.py` with no hidden hand-entered result tables.

Baselines:

1. `raw_trajectory_knn`: nearest-prototype imitation on raw keypoint trajectories.
2. `body_normalized_knn`: scale and handedness normalized nearest-prototype baseline.
3. `velocity_contact_classifier`: trajectory dynamics plus inferred contact timing.
4. `object_affordance_classifier`: object-state and affordance features without morphology factorization.
5. `style_invariant_logistic`: a lightweight learned classifier trained on morphology-normalized features.

Proposed mechanism:

`physical_intent_disambiguation`: factor body scale, handedness, style speed, object frame, contact timing, and affordance-consistent object displacement before classifying latent physical intent and selecting the robot action.

Upper bound:

`oracle_intent_upper_bound`: uses ground-truth intent and action parameters to estimate remaining noise and actuation limits.

## Metrics

Report seed-level and aggregate metrics:

1. Intent accuracy.
2. Robot action success.
3. Action-parameter error.
4. Calibration error.
5. Morphology leakage from method features.
6. Paired success-rate differences versus the strongest non-oracle baseline.

## Ablations

Evaluate proposed variants on the combined hard split:

1. Full mechanism.
2. Minus body-scale normalization.
3. Minus handedness mirroring.
4. Minus object-affordance constraints.
5. Minus contact-timing features.
6. Minus style-speed normalization.
7. Minus uncertainty/abstention gate.

## Stress Tests

Sweep independent stress levels for morphology scale gap, style exaggeration, occlusion/noise, affordance ambiguity, and partial observation. Compare the proposed method against the strongest baselines and the oracle upper bound.

## Paper Update

Rewrite the paper as a compact ICLR-style submission-hardening report:

1. State the method and the honest evidence ceiling.
2. Describe benchmark generation and splits in reproducible detail.
3. Present main results, ablations, stress tests, and negative cases.
4. Include hostile related-work pressure and limitations.
5. Use the terminal decision supported by the evidence, not the desired outcome.

## Delivery Checklist

1. Run all experiments from scratch.
2. Produce CSVs and figures in the repo.
3. Compile the PDF and copy only `80.pdf` to `C:\Users\wangz\Downloads`.
4. Update local status files and root pool reports.
5. Commit changes and push the matching public GitHub repository.

