# CLAUDE.md

This file gives Claude Code context when working in this repository. Read this before making changes, adding files, or suggesting new scope.

## Project Summary

We are modernizing a prior research paper that detected adversarial attacks on
images by analyzing LIME explanation heatmaps (distribution-based metrics: KL
divergence, Gini coefficient, spectral/FFT features) and a CNN trained directly
on those heatmaps. That paper's "attacks" were mostly image *corruptions*
(Gaussian, Poisson, speckle noise) rather than true adversarial attacks, and
LIME is now known to be unstable (sensitive to superpixel segmentation,
non-deterministic across runs).

This project rebuilds that idea with modern components and compares two
detection philosophies side by side:

- **Pipeline 1 (explanation-based):** CNN classifier + Grad-CAM (and
  optionally other explainers, e.g. Integrated Gradients). Detect attacks by
  measuring how the explanation heatmap's statistical properties differ
  between clean and adversarial images.
- **Pipeline 2 (representation-based):** DINOv2 (ViT-based self-supervised
  encoder). No explanation step. Detect attacks by measuring how far an
  image's embedding sits from the distribution of clean embeddings
  (distance-based scoring, e.g. Mahalanobis or nearest-neighbor).

Both pipelines are evaluated against the same adversarial attacks, using the
same metrics, so their strengths/weaknesses can be fairly compared.

**Course context:** this is a Bloomberg–Columbia summer AI-in-finance course
project, guided by Gary (Bloomberg) and Miao (teaching assistant). It is
NOT a from-scratch research exploration — direction has already been
negotiated and confirmed with them. Do not propose new research directions
without being asked; implement the agreed plan below.

## Pipeline (standard flow for any experiment)

```
Dataset -> Model/Encoder -> Adversarial Attack -> Explainability Method (Pipeline 1 only)
        -> Feature Extraction -> Detector -> Evaluation
```

Pipeline 2 skips the "Explainability Method" step — detection happens
directly on the encoder's raw embedding output.

## Repository Layout

```
configs/        Experiment configs (dataset, attack params, model, eval settings)
data/           Datasets (not committed)
attacks/        Adversarial attack implementations — SHARED across both pipelines
encoders/       Vision backbones (CNN for Pipeline 1, DINOv2 for Pipeline 2)
explainers/     Grad-CAM, etc. — Pipeline 1 only
detectors/      Detection models built on extracted features (heatmap metrics
                or embedding distances) — used by both pipelines
analysis/       Evaluation metrics, plots, dimensionality reduction — SHARED,
                must be used identically by both pipelines for fair comparison
experiments/    Entry-point scripts, one per (pipeline, dataset) combination
results/        figures/, tables/, logs/
notebooks/      Exploratory only — nothing here should be load-bearing for
                final results
tests/
```

**Critical:** `attacks/` and `analysis/` are shared infrastructure. Do not
fork or duplicate attack-generation or metric-computation logic between
pipelines — both pipelines must be attacked and scored identically, or the
comparison between them is not valid.

## Team Ownership (current)

- **Eric:** `encoders/cnn.py`, `explainers/gradcam.py`, Pipeline 1 experiments
- **Teju and Shayl:** `encoders/dinov2.py`, `detectors/` (distance-based scoring), `analysis/` (metrics),
  Pipeline 2 evaluation glue

`attacks/` ownership: [TBD — confirm with team; likely Eric's since it needs
a working classifier with gradients, but not yet explicitly assigned]

## Data / Interface Conventions

[TBD — must be confirmed across the team before pipelines are connected]

- Image size / preprocessing: ?
- Tensor shape and format expected in/out of `attacks/`: ?
- Batch format: ?
- Which dataset(s) first: CIFAR-10 is the current default (cheaper than
  ImageNet-1K) — confirm before assuming otherwise

Do not assume a format — check `configs/` or ask before writing code that
depends on one.

## Evaluation Standard

Use these metrics for every detector, computed identically across both
pipelines:

- **AUC** (primary headline metric)
- **TPR at fixed FPR** (1% and 5%) — more practical than raw accuracy
- Evaluate **only on successful attacks** (clean prediction correct AND
  adversarial prediction incorrect) — do not include failed attack attempts
  in detection scoring

This protocol is adopted from ViTGuard (Sun et al., ACSAC 2024) rather than
invented from scratch.

## Attacks In Scope

- FGSM, PGD (already implemented/working — see Shayl's earlier ViT pipeline
  for a reference implementation of attack-generation + PSNR-based
  imperceptibility checking)
- APGD, CW (stretch)
- One patch attack (stretch)
- One transfer attack (stretch)
- Gaussian noise: NOT an attack — use as a labeled **control** to test
  whether detectors distinguish real attacks from mere corruption. This is
  a deliberate, important experiment, not an oversight.

## Explicit Scope Boundaries

The following have been discussed and intentionally excluded or deferred.
Do not add them without asking the group first:

- CLIP as an encoder — stretch goal only, not core
- Genuinely different architectures beyond CNN/DINOv2 (e.g. Swin, CaiT) —
  out of scope
- Adaptive attacks (attacks designed with knowledge of the detector) —
  out of scope; this alone is a multi-week research direction
- More than one detector method per pipeline as a starting point — build
  one working detector per pipeline first (e.g. logistic regression or
  distance-based scoring); only add more (KNN, MLP, Mahalanobis) if time
  remains after the core comparison works
- Multiple datasets simultaneously — pick one (CIFAR-10 default) and get it
  fully working before adding a second

**If asked to suggest "next steps" or "what else could we add," prioritize
finishing the existing scope over proposing new experiments.** This team has
a hard deadline.

## Coding Style (course requirement — non-negotiable)

- OOP design: base classes with inheritance where models/detectors share
  components (e.g. a shared base class for detector types)
- Hardcoded values (except in test scripts) must be named global constants,
  ALL_CAPS_WITH_UNDERSCORES, e.g. `EPSILON_DEFAULT = 4/255`
- Every class and function needs docstrings specifying input/output types
- Raw data as CSV files, source documented in a README
- Configs (not hardcoded values) drive experiment variation — see `configs/`

## Deadline

Final report, code, video (15 min), and poster materials due **Sep 4, 2026,
noon**. Code must be clean and documented by then — do not defer refactoring
to "later," there is limited buffer. When code stabilizes, refactor promptly
rather than continuing to add features on top of unstructured scripts.

## Key References

- Original paper this project modernizes: LIME-based adversarial detection
  (see `docs/` or ask team for the PDF)
- ViTGuard (Sun et al., ACSAC 2024) — evaluation protocol source
- Lee et al. 2018, "A Simple Unified Framework for Detecting OOD Samples and
  Adversarial Attacks" — Mahalanobis distance baseline for Pipeline 2
- Feinman et al. 2017, "Detecting Adversarial Samples from Artifacts" —
  earliest internals-based detection baseline
