# Claims

- Mechanism claim: factoring body normalization, style-invariant evidence, object affordances, contact cues, and physical action geometry can improve robot intent inference from human keypoint video under morphology/style/affordance shift.
- Evidence claim: the v4 local benchmark shows `physical_intent_disambiguation` reaches `0.779 +/- 0.028` action success on `combined_hard_shift`, beating the strongest non-oracle baseline at `0.605 +/- 0.043`.
- Leakage claim: raw video features leak morphology at `0.782` probe accuracy, while the factored representation reduces leakage to `0.578`.
- Ablation claim: object-affordance evidence is important; contact timing and uncertainty gating are not validated as necessary by this local benchmark.
- Scope claim: results support a promising local mechanism and justify `STRONG_REVISE`, not ICLR-main submission.
- Unsupported claim explicitly avoided: no claim of SOTA robot performance, real-video generalization, or hardware deployment.

