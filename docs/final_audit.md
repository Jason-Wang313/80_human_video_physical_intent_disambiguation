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

## 2026-06-15 Continuation Audit

1. Plan-first requirement: satisfied by `docs/paper80_iclr_submission_execution_plan_20260615.md` before the evidence gate was rerun.
2. Code gate: `python -m py_compile src/run_experiment.py` passed.
3. Experiment gate: `python src/run_experiment.py` completed with terminal recommendation STRONG_REVISE.
4. CSV integrity gate: audited 10,290 `rollouts.csv` rows, 1,470 `dataset_summary.csv` rows, 210 `morphology_leakage.csv` rows, 245 `raw_seed_metrics.csv` rows, 205 `metrics.csv` rows, 75 `pairwise_stats.csv` rows, 2,058 `ablation_rollouts.csv` rows, 7 `ablation_metrics.csv` rows, 20,160 `stress_sweep_raw.csv` rows, 120 `stress_sweep.csv` rows, and 16 `negative_cases.csv` rows.
5. Coverage gate: seeds 0 through 6, seven main methods, five evaluation splits, seven ablations, four stress axes, and six stress levels are present.
6. Decisive split: on `combined_hard_shift`, `physical_intent_disambiguation` reaches 0.77891 +/- 0.02828 action success, while `object_affordance_classifier` reaches 0.60544 +/- 0.04303 and `style_invariant_logistic` reaches 0.48639 +/- 0.05837.
7. Paired statistics: the action-success gain over `object_affordance_classifier` is 0.17347 +/- 0.02211 with 7/7 better seeds; the gain over `style_invariant_logistic` is 0.29252 +/- 0.06498 with 7/7 better seeds.
8. Morphology-leakage gate: raw trajectory features leak morphology at 0.78231, while the factored method is 0.57823.
9. Ablation gate: object-affordance removal drops action success from 0.77891 to 0.66327, supporting the central object/factoring mechanism; contact timing and uncertainty gating remain neutral and should not be overclaimed.
10. Stress gate: at maximum combined stress, `physical_intent_disambiguation` reaches 0.71429 +/- 0.04518 versus 0.51786 +/- 0.08139 for `object_affordance_classifier` and 0.47619 +/- 0.05872 for `style_invariant_logistic`.
11. PDF gate: `paper/main.pdf` rebuilt after BibTeX-author and float-placement cleanup, then copied to `C:/Users/wangz/Downloads/80.pdf`.
12. Artifact gate: `C:/Users/wangz/Downloads/80.pdf` SHA256 is `C3BA4F78792B565CEC2DC80658E3AF551D0F792EABF66F0C0783039A2684B0DC`; `C:/Users/wangz/Desktop/80.pdf` is absent.
13. Final decision: STRONG_REVISE. The local evidence is promising and reproducible, but the artifact is not ICLR-main-ready without real human-video, robot hardware, external learned-baseline, or recognized high-fidelity benchmark validation.
