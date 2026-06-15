# Paper 80 ICLR Submission-Readiness Execution Plan - 2026-06-15

## Objective

Re-audit Paper 80, `human_video_physical_intent_disambiguation`, as a real ICLR-main-target candidate without inflating the claim beyond the evidence. The paper may remain `STRONG_REVISE` if the local keypoint-video benchmark is reproducible and the proposed method decisively beats strong local baselines. It cannot be marked ICLR-main-ready without real human-video, robot hardware, recognized high-fidelity benchmark, or external learned video-to-robot baseline evidence.

## Evidence Gate

1. Verify the runner with `python -m py_compile src/run_experiment.py`.
2. Re-run `python src/run_experiment.py` once, using the existing lightweight deterministic benchmark and without reducing seeds, baselines, ablations, or stress sweeps.
3. Audit CSV integrity and expected scale:
   - `rollouts.csv`: 10,290 main rollout rows.
   - `raw_seed_metrics.csv`: seven seeds across all main methods/splits.
   - `metrics.csv` and `pairwise_stats.csv`: aggregate and paired uncertainty for every split.
   - `ablation_rollouts.csv`: 2,058 rollout rows.
   - `ablation_metrics.csv`: seven ablation rows.
   - `stress_sweep_raw.csv`: 20,160 rollout rows.
   - `stress_sweep.csv`: stress-axis/method/level summaries.
   - `morphology_leakage.csv`, `dataset_summary.csv`, and `negative_cases.csv`: diagnostic coverage.
4. Confirm seeds 0 through 6, all seven main methods, all five evaluation splits, all seven ablations, and the four stress axes are present.

## Decision Criteria

Keep `STRONG_REVISE` only if all of the following still hold:

1. On `combined_hard_shift`, `physical_intent_disambiguation` beats the strongest non-oracle baseline by a practically meaningful margin.
2. Paired action-success gain over `object_affordance_classifier` and `style_invariant_logistic` is positive with uncertainty reported over seeds.
3. Morphology leakage is reduced relative to raw trajectory features while preserving higher action success than object-only and style-invariant baselines.
4. Ablations support at least the central object-affordance/factoring mechanism, while neutral components are reported honestly.
5. Stress sweeps do not reveal a decisive reversal against the strongest non-oracle baseline at maximum combined stress.
6. The paper explicitly states that local generated evidence is insufficient for ICLR-main submission readiness.

Downgrade to `KILL_ARCHIVE` if the proposed method no longer clears strong baselines, if ablations contradict the mechanism, if stress reverses the main claim, if reproduced CSVs differ materially from the manuscript, or if the artifact cannot be rebuilt cleanly.

Do not upgrade to ICLR-main-ready unless new external/real-video/robot/high-fidelity evidence is actually present in the repository.

## Artifact Gate

1. Rebuild the PDF with `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`.
2. Fix recoverable LaTeX/BibTeX issues, including placeholder bibliography warnings or fragile float placement, without changing the empirical claim.
3. Copy only the canonical numbered PDF to `C:/Users/wangz/Downloads/80.pdf`.
4. Confirm `C:/Users/wangz/Desktop/80.pdf` is absent.
5. Record the Downloads PDF SHA256.

## Documentation And Repo Gate

1. Update child status, plan, final audit, hostile reviewer response, attack log, submission readiness docs, and version log with the continuation result.
2. Update root `GLOBAL_POOL_STATUS.md`, `BATCH_STATUS.md`, `SUBMISSION_STATUS.md`, `MASTER_REPORT.md`, and `MASTER_SUBMISSION_REPORT.md` so the continuation audit is current through Paper 80.
3. Verify the public GitHub repository URL and visibility.
4. Commit and push the Paper 80 repository.
5. Verify local `HEAD` equals `origin/main` and the worktree is clean before moving to Paper 81.

## RAM Discipline

Run one Paper 80 experiment process at a time and keep the existing row counts/seeds/baselines intact. Do not trade away experiment quality to save memory; use the repo's deterministic lightweight benchmark as designed.
