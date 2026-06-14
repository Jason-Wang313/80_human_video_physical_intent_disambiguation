# 80 Human Video Physical Intent Disambiguation

Submission-hardening version: v4

Terminal decision: STRONG_REVISE for ICLR main conference.

Paper 80 was rebuilt from a v3 synthetic archive into a local keypoint-video benchmark for learning robot actions from human demonstrations under morphology, style, object-affordance, contact, occlusion, and noise shifts. The runner implements raw trajectory imitation, body-normalized kNN, velocity/contact classification, object-affordance classification, a style-invariant logistic classifier, the proposed factored physical-intent disambiguator, ablations, stress sweeps, morphology-leakage probes, negative cases, and figures.

The local evidence is promising but not final. On the decisive `combined_hard_shift` split, `physical_intent_disambiguation` reaches `0.779 +/- 0.028` robot action success, outperforming `object_affordance_classifier` at `0.605 +/- 0.043`, `style_invariant_logistic` at `0.486 +/- 0.058`, and `body_normalized_knn` at `0.303 +/- 0.044`. Morphology leakage drops from `0.782` for raw video features to `0.578` for the factored representation. The paired action-success gain over the strongest non-oracle baseline is `0.173 +/- 0.022`.

This is still not ICLR-main submission-ready because the evidence is a generated local benchmark, not real human video, hardware validation, or a recognized high-fidelity robotics suite.

## Reproduce Evidence

```powershell
python src\run_experiment.py
```

This writes:

- `results/rollouts.csv` with 10,290 main rollout rows.
- `results/dataset_summary.csv`.
- `results/morphology_leakage.csv`.
- `results/raw_seed_metrics.csv`.
- `results/metrics.csv`.
- `results/pairwise_stats.csv`.
- `results/ablation_rollouts.csv` and `results/ablation_metrics.csv`.
- `results/stress_sweep_raw.csv` and `results/stress_sweep.csv`.
- `results/negative_cases.csv`.
- Figures under `figures/`.

## Rebuild PDF

```powershell
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Canonical local PDF: `C:/Users/wangz/Downloads/80.pdf`

No visible Desktop PDF is required or produced.
