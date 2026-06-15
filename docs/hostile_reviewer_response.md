# Hostile Reviewer Response

Paper: 80 Human Video Physical Intent Disambiguation

## Strongest Technical Threats
- Estimating Articulated Human Motion with Covariance Scaled Sampling (2003)
- CoRI: Communication of Robot Intent for Physical Human-Robot Interaction (2025)
- Human motion intent learning based motion assistance control for a wearable exoskeleton (2018)
- A Bio-Inspired Mechanism for Learning Robot Motion From Mirrored Human Demonstrations (2022)
- A probabilistic model of human motion and navigation intent for mobile robot path planning (2009)
- REPLAY: Robot Embodiment via Intent-aware Policy Imitation by Replicating Human Demonstrations From Video (2025)
- Auction-Based Task Allocation and Motion Planning for Multi-Robot Systems with Human Supervision (2023)
- Video Human Action Recognition Based on Motion-Tempo Learning and Feedback Attention (2025)

## ICLR Main Response
A hostile ICLR reviewer would still be correct to reject this as a final main-conference submission because the v4 evidence is generated local keypoint-video data, not real human-video or robot validation. However, the v4 rebuild is no longer a template probability scaffold: it implements paper-specific baselines, a factored physical-intent method, paired statistics, ablations, stress tests, leakage probes, and figures.

## Honest Action
The paper is marked `STRONG_REVISE`. The local result is promising enough to continue, but not enough to submit.

## What Would Be Needed To Submit
- Real human-video-to-robot experiments or a recognized high-fidelity benchmark.
- External learned video-to-robot baselines.
- Qualitative real rollouts or videos.
- Manual full-paper related-work audit.
- A narrower claim around the components that the ablations actually validate.

## 2026-06-15 Continuation Response

The continuation audit reran the benchmark and still gives a meaningful local positive result:

- `physical_intent_disambiguation`: 0.77891 +/- 0.02828 action success on `combined_hard_shift`.
- `object_affordance_classifier`: 0.60544 +/- 0.04303 action success.
- `style_invariant_logistic`: 0.48639 +/- 0.05837 action success.
- Paired action-success gain over `object_affordance_classifier`: 0.17347 +/- 0.02211, better in 7/7 seeds.
- Morphology leakage drops from 0.78231 for raw trajectory features to 0.57823 for the factored method.
- At maximum combined stress, the proposed method reaches 0.71429 +/- 0.04518 versus 0.51786 +/- 0.08139 for object-affordance classification.

The hostile reviewer still wins on submission readiness because all of this is generated local keypoint-video evidence. The honest action remains STRONG_REVISE, not ICLR-main-ready.
