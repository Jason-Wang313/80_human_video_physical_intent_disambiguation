# ICLR Main Gate

Paper: 80 human_video_physical_intent_disambiguation

Previous v3 decision: KILL_ARCHIVE

Current v4 gate verdict: STRONG_REVISE

Evidence digest: regenerated in v4 from the local keypoint-video benchmark.

Positive evidence:
- Implemented benchmark with keypoint trajectories, object state, contact timing, morphology, style, affordances, occlusion, and noise.
- Implemented baselines: raw trajectory kNN, body-normalized kNN, velocity/contact classifier, object-affordance classifier, style-invariant logistic classifier, proposed factored method, and oracle upper bound.
- Seven seeds, 10,290 main rollout rows, 2,058 ablation rollout rows, and 20,160 stress-sweep rows.
- On `combined_hard_shift`, proposed action success is `0.779 +/- 0.028` versus `0.605 +/- 0.043` for the strongest non-oracle baseline.
- Morphology leakage falls from `0.782` for raw video features to `0.578` for the factored representation.

Remaining blockers:
- No real human-video dataset.
- No robot hardware validation.
- No recognized high-fidelity simulator benchmark.
- No external learned video-to-robot baseline.
- Manual related-work synthesis remains thinner than a final ICLR submission requires.

Gate action: continue as `STRONG_REVISE`; do not claim ICLR-main readiness until external empirical validation exists.

