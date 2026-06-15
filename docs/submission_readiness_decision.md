# Submission Readiness Decision

Decision: STRONG_REVISE

ICLR main-conference readiness: NO.

Reason: The v4 rebuild now contains an implemented local keypoint-video benchmark, strong local baselines, seven seeds, paired comparisons, ablations, stress sweeps, morphology-leakage probes, negative cases, figures, and a reproducible PDF. On `combined_hard_shift`, `physical_intent_disambiguation` reaches `0.779 +/- 0.028` action success versus `0.605 +/- 0.043` for the strongest non-oracle baseline, with paired gain `0.173 +/- 0.022`.

The paper is not yet ready for ICLR main because the evidence is generated local keypoint-video data, not real human-video demonstrations, hardware rollouts, or a recognized high-fidelity robotics benchmark.

Honest terminal action: keep as `STRONG_REVISE`; do not submit to ICLR main without external empirical validation.

Revival condition: validate the factored representation on real human-video-to-robot data or a recognized simulator suite, add learned visual encoders and external baselines, then rewrite as a full empirical submission.

## 2026-06-15 Continuation Decision

Decision: STRONG_REVISE.

ICLR main-conference readiness: NO.

Reason: the full deterministic benchmark was rerun and the positive local evidence reproduces. On `combined_hard_shift`, `physical_intent_disambiguation` reaches 0.77891 +/- 0.02828 action success versus 0.60544 +/- 0.04303 for `object_affordance_classifier`, with paired gain 0.17347 +/- 0.02211 and 7/7 better seeds. Morphology leakage falls from 0.78231 for raw trajectories to 0.57823 for the factored method, and maximum combined stress remains favorable at 0.71429 versus 0.51786 for object-affordance classification.

Blocker: the evidence is still generated local keypoint-video data with no real human-video demonstrations, robot hardware validation, recognized high-fidelity benchmark, or external learned video-to-robot baseline. This blocks ICLR-main readiness even though the local mechanism is promising.
