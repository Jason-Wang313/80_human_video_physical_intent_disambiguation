# Submission Version Log

## v1 - Generated Draft
- Original continuation-batch generated paper and toy single-seed experiment.

## v2 - Submission Hardening
- Added hostile reviewer attack log and response docs.
- Replaced the toy experiment with seven-seed metrics, stronger baselines, ablations, stress tests, and negative cases.
- Narrowed claims to synthetic diagnostic evidence.
- Recompiled canonical PDF at `C:/Users/wangz/Downloads/80.pdf`.
- Terminal decision: WORKSHOP_ONLY.

## v3 - ICLR Main Gate Archive
- Applied the stricter ICLR-main-conference standard.
- Re-read local paper, docs, experiments, prior-work artifacts, PDF state, and repo state.
- Determined that missing real-robot/high-fidelity evidence, template-generated experiments, and unresolved novelty threats were not recoverable from local artifacts.
- Recompiled the canonical PDF with `Submission-hardening version: v3`.
- Terminal decision: KILL_ARCHIVE.

## v4 - Physical Intent Rebuild
- Added `docs/paper80_rebuild_plan.md` before executing changes.
- Replaced the scalar probability scaffold with an implemented keypoint-video benchmark for morphology, style, affordance, contact, occlusion, and noise shifts.
- Implemented raw trajectory kNN, body-normalized kNN, velocity/contact classification, object-affordance classification, style-invariant logistic classification, factored physical-intent disambiguation, and oracle upper bound.
- Ran seven seeds, 10,290 main rollout rows, 2,058 ablation rollout rows, and 20,160 stress-sweep rows.
- Produced figures, paired statistics, morphology-leakage probes, ablations, stress sweeps, and negative cases.
- Terminal decision: STRONG_REVISE.

## v4 Continuation Audit - 2026-06-15

- Added `docs/paper80_iclr_submission_execution_plan_20260615.md` before rerunning the evidence gate.
- Recompiled and reran the full deterministic benchmark without reducing seeds, baselines, ablations, or stress sweeps.
- Rechecked CSV integrity, seed/method/split/ablation/stress coverage, paired statistics, morphology leakage, ablations, stress behavior, PDF logs, Downloads-only placement, Desktop exclusion, and public GitHub status.
- Cleaned BibTeX placeholder entries by adding explicit authors and changed fragile `[h]` floats to `[tbp]` before rebuilding the PDF.
- Verified `C:/Users/wangz/Downloads/80.pdf` SHA256 `C3BA4F78792B565CEC2DC80658E3AF551D0F792EABF66F0C0783039A2684B0DC`.
- Terminal decision remains: STRONG_REVISE.
