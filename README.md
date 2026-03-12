# Compositional vs. Conceptual Novelty in Text-to-Image Models

> **Do diffusion models truly imagine, or do they just dream?**
> 
> An empirical study investigating whether Stable Diffusion XL produces genuinely novel visual concepts — or merely recombines patterns from its training data.

---

## Overview

This repository contains the code and evaluation framework for the paper **"Compositional vs. Conceptual Novelty in Text-to-Image Models"**, which draws a parallel between human dreaming and the generative behavior of diffusion models. Just as dreams reconstruct familiar experiences into novel configurations, diffusion models appear to generate "creative" outputs through structured recombination within a learned representation space — rather than true conceptual invention.

The study evaluates 120 generated images across three prompt abstraction levels using CLIP-based alignment scores, embedding-based novelty metrics, and intra-prompt diversity measures, with full statistical analysis.

---

## Research Question

> *To what extent do diffusion-based generative models produce genuinely novel visual concepts, and how similar is this process to the compositional and associative mechanisms observed in human dreaming?*

---

## Key Findings

- As prompt abstraction increases (Level 1 → Level 3), **novelty increases** while **CLIP alignment decreases** — the model drifts further from its training distribution but loses semantic fidelity.
- Novelty scores (0.46–0.49) indicate images are anchored to the learned manifold while exhibiting moderate distributional deviation.
- **One-way ANOVA** confirmed statistically significant differences across all three levels for all metrics (p < 0.05).
- **Large Cohen's d effect sizes** were observed between Level 1 and Level 3 for both CLIP score and approximate novelty.
- Strong positive correlation between prompt diversity and novelty; weak correlation between novelty and CLIP score — supporting that creativity and semantic alignment are partially independent.
- Bootstrap estimates confirmed results are stable and not sensitive to the specific sample.

---

## Prompt Abstraction Levels

| Level | Type | Description |
|---|---|---|
| **Level 1** | Direct | Realistic, grounded scenes (in-distribution) |
| **Level 2** | Compositional | Surreal hybrid imagery combining multiple known concepts |
| **Level 3** | Abstract | Philosophically or conceptually non-concrete descriptions |

Each level contains 5 unique prompts. 8 images are generated per prompt (different random seeds), yielding **120 images total** (40 per level).

---

## Methodology

### Pipeline

```
Prompts (3 levels × 5 prompts)
        ↓
Image Generation — Stable Diffusion XL (8 seeds/prompt)
        ↓
Feature Extraction — CLIP ViT-Large/14
        ↓
Metrics:
  • CLIP Score       → image-text semantic alignment
  • Approximate Novelty  → deviation from MS-COCO reference distribution
  • Prompt Diversity     → intra-prompt pairwise embedding distances
        ↓
Statistical Analysis — ANOVA, Tukey HSD, Cohen's d, Bootstrap CIs
```

### Models & Datasets

| Component | Choice | Reason |
|---|---|---|
| Text-to-Image | `stabilityai/stable-diffusion-xl-base-1.0` | Open-source, reproducible, widely adopted |
| CLIP | `openai/clip-vit-large-patch14` | Standard for SDXL evaluation; used during SDXL training |
| Reference baseline | MS-COCO (2,000 images) | High-resolution, diverse, photorealistic — appropriate baseline for SDXL outputs |

> **Note:** CIFAR-10 (32×32 low-res) is an *inappropriate* baseline for SDXL outputs as it artificially inflates novelty scores. MS-COCO is the correct choice used here.

---

## Repository Structure

```
├── image_novelty_evaluation_fixed.ipynb   # Main evaluation notebook
├── generated_images/                      # Generated images (120 total)
├── results_boxplots.png                   # Metric distribution across levels
├── results_violins.png                    # Violin plots with quartiles
├── results_bar_ci.png                     # Mean scores with 95% CI bars
├── results_trends.png                     # Metric trend lines across levels
├── results_correlation.png               # Metric correlation heatmap
└── README.md
```

---

## Setup & Requirements

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (strongly recommended; SDXL generation is slow on CPU)

### Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate datasets
pip install scikit-learn scipy seaborn matplotlib pandas statsmodels
```

### Running the Notebook

```bash
jupyter notebook image_novelty_evaluation_fixed.ipynb
```

The notebook will:
1. Load Stable Diffusion XL from HuggingFace Hub
2. Generate 120 images (3 levels × 5 prompts × 8 seeds)
3. Load CLIP ViT-Large/14
4. Stream 2,000 MS-COCO images to build a reference embedding baseline
5. Compute CLIP scores, novelty, and diversity metrics
6. Run ANOVA, Tukey HSD, Cohen's d, bootstrap, and correlation analyses
7. Save all result plots to the working directory

> ⚠️ **Compute note:** Full image generation requires a GPU with at least 10GB VRAM. Estimated runtime: ~1–2 hours on an A100 / ~3–5 hours on a consumer GPU.

---

## Statistical Analysis Summary

| Test | Purpose |
|---|---|
| One-way ANOVA | Test whether abstraction level significantly affects each metric |
| Tukey HSD post-hoc | Pairwise comparisons between levels |
| Cohen's d | Effect size magnitude between level pairs |
| 95% Confidence Intervals | Parametric and bootstrap-estimated |
| Bootstrap resampling | Stability check for mean estimates |
| Pearson Correlation | Relationship between CLIP score, novelty, and diversity |

---

## Hypotheses Tested

**H1 — Generative modeling as generalization of experience:** The model only produces what can be expressed through concepts encountered during training, analogous to how humans only dream of what they have previously experienced.

**H2 — The limits of novelty: combination, not invention:** The model can combine familiar elements into new configurations, but cannot invent elements outside its training experience (e.g., "cat + chair" → "cat on a chair," not an entirely new entity).

**H3 — Direct recall vs. generation:** In some cases the model may retrieve near-identical images from training data, analogous to dreaming of a scene experienced exactly as seen before.

---

## Citation

If you use this work, please cite:

```bibtex
@article{novelty2024,
  title   = {Compositional vs. Conceptual Novelty in Text-to-Image Models},
  year    = {2024},
  note    = {Empirical study using Stable Diffusion XL and CLIP-based evaluation}
}
```

---

## References

1. Liu et al. (2022) — Compositional Visual Generation with Composable Diffusion Models
2. Wu et al. (2024) — ConceptMix: A Compositional Image Generation Benchmark
3. Ramaswamy et al. (2024) — Quantitative Measures of Task-Oriented Creativity in Generative Vision Models
4. Deperrois et al. (2023) — Learning beyond sensations: How dreams organize neuronal representations
5. Stability AI — Stable Diffusion XL
6. Radford et al. — CLIP: Learning Transferable Visual Models From Natural Language Supervision
7. Lin et al. — Microsoft COCO: Common Objects in Context

---

## License

This project is released for academic and research purposes.
