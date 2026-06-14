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

