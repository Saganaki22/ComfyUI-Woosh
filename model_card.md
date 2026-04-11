---
license: apache-2.0
tags:
  - audio
  - text-to-audio
  - video-to-audio
  - sound-effects
  - flow-matching
  - diffusion
  - comfyui
library_name: woosh
---

# Woosh — Sound Effect Generative Models

Inference code and open weights for sound effect generative models developed at Sony AI.

[![GitHub](https://img.shields.io/badge/GitHub-SonyResearch%2FWoosh-black)](https://github.com/SonyResearch/Woosh)
[![ComfyUI Node](https://img.shields.io/badge/ComfyUI-ComfyUI--Woosh-blue)](https://github.com/Saganaki22/ComfyUI-Woosh)
[![arXiv](https://img.shields.io/badge/arXiv-2502.07359-b31b1b)](https://arxiv.org/abs/2502.07359)

## Models

| Model | Task | Steps | CFG | Description |
|-------|------|-------|-----|-------------|
| **Woosh-Flow** | Text-to-Audio | 50 | 4.5 | Base model, best quality |
| **Woosh-DFlow** | Text-to-Audio | 4 | 1.0 | Distilled Flow, fast generation |
| **Woosh-VFlow** | Video-to-Audio | 50 | 4.5 | Base video-to-audio model |
| **Woosh-DVFlow** | Video-to-Audio | 4 | 1.0 | Distilled VFlow, fast video-to-audio |

### Components

- **Woosh-AE** — High-quality latent encoder/decoder. Provides latents for generative modeling and decodes audio from generated latents.
- **Woosh-CLAP (TextConditionerA/V)** — Multimodal text-audio alignment model. Provides token latents for diffusion model conditioning. TextConditionerA for T2A, TextConditionerV for V2A.
- **Woosh-Flow / Woosh-DFlow** — Original and distilled LDMs for text-to-audio generation.
- **Woosh-VFlow** — Multimodal LDM generating audio from video with optional text prompts.

## ComfyUI Nodes

Use these models in [ComfyUI](https://github.com/comfyanonymous/ComfyUI) with [ComfyUI-Woosh](https://github.com/Saganaki22/ComfyUI-Woosh):

```bash
# Via ComfyUI Manager — search "Woosh" and click Install
# Or manually:
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/ComfyUI-Woosh.git
pip install -r ComfyUI-Woosh/requirements.txt
```

Place downloaded model folders in `ComfyUI/models/woosh/`. See the [ComfyUI-Woosh README](https://github.com/Saganaki22/ComfyUI-Woosh) for full setup and workflow examples.

> **Note:** Set the Woosh TextConditioning node to **T2A** for Flow/DFlow models and **V2A** for VFlow/DVFlow models.

## Inference

See the [official Woosh repository](https://github.com/SonyResearch/Woosh) for standalone inference code and training details.

## VRAM Requirements

| Model | VRAM (Approx) |
|-------|---------------|
| Flow / VFlow | ~8-12 GB |
| DFlow / DVFlow | ~4-6 GB |
| With CPU offload | ~2-4 GB |

## Citation

```bibtex
@article{saghibakshi2025woosh,
      title={Woosh: Enhancing Text-to-Audio Generation with Flow Matching and FlowMap Distillation},
      author={Saghibakshi, Ali and Bakshi, Soroosh and Tagliasacchi, Antonio and Wang, Shaojie and Choi, Jongmin and Kawakami, Kazuhiro and Gu, Yuxuan},
      journal={arXiv preprint arXiv:2502.07359},
      year={2025}
}
```

## License

Apache 2.0
