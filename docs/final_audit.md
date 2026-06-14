# Final Audit

1. Chosen thesis: Human Video Physical Intent Disambiguation explores `Separate human intent from morphology-specific motion artifacts in video demonstrations.` for learning robots from human video.
2. ICLR-main decision: STRONG_REVISE.
3. Submission-hardening version: v4.
4. Reason: the v4 rebuild adds implemented local evidence and beats strong local baselines, but still lacks real human-video, hardware, or recognized high-fidelity benchmark validation.
5. Decisive result: on `combined_hard_shift`, `physical_intent_disambiguation` reaches `0.779 +/- 0.028` action success versus `0.605 +/- 0.043` for `object_affordance_classifier`.
6. Paired result: action-success gain over the strongest non-oracle baseline is `0.173 +/- 0.022`.
7. Morphology leakage: raw video features score `0.782`; the factored representation scores `0.578`.
8. Caveat: object-affordance ablation is important, but contact timing and uncertainty gating are not validated by the local benchmark.
9. Closest hostile prior work: see `docs/hostile_prior_work.md`, `docs/hostile_prior_work_100_cards.csv`, and `docs/hostile_reviewer_response.md`.
10. Reproducibility: `python src\run_experiment.py` regenerates metrics and figures.
11. Claim-validity status: promising local mechanism; not submission-ready until external validation is added.
12. Exact Downloads PDF path: `C:/Users/wangz/Downloads/80.pdf`
13. GitHub URL: https://github.com/Jason-Wang313/80_human_video_physical_intent_disambiguation
14. Confirmation: no visible Desktop copy was requested or made.

