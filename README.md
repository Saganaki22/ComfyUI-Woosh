# ComfyUI-Woosh

**Sound effect generation nodes for ComfyUI** — Text-to-audio and video-to-audio using Sony AI's Woosh foundation model.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Woosh-yellow)](https://huggingface.co/drbaph/Woosh)
[![arXiv](https://img.shields.io/badge/arXiv-2502.07359-b31b1b)](https://arxiv.org/abs/2502.07359)
[![Original Repo](https://img.shields.io/badge/GitHub-SonyResearch%2FWoosh-black)](https://github.com/SonyResearch/Woosh)
[![ComfyUI Node](https://img.shields.io/badge/ComfyUI-ComfyUI--Woosh-blue)](https://github.com/Saganaki22/ComfyUI-Woosh)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

<img width="1878" height="1261" alt="Screenshot 2026-04-12 013347" src="https://github.com/user-attachments/assets/60566958-581d-4b77-aa2f-3b39e3b521a4" />



## Features

- **Text-to-Audio** — Generate sound effects from text descriptions (Flow/DFlow)
- **Video-to-Audio** — Generate audio from video frames (VFlow/DVFlow)
- **Distilled Models** — DFlow/DVFlow for fast 4-step generation
- **Dynamic VRAM** — GPU<->CPU offload via ComfyUI's ModelPatcher
- **Force Offload** — Throw model from GPU+CPU RAM after generation, reloads from disk next run
- **Video Output** — V2A outputs video frames directly for VideoCombine
- **No pip install** — Woosh library bundled, zero risk to your torch environment



https://github.com/user-attachments/assets/243eef4f-8146-465d-b579-217dff4baa2b



## Installation

### ComfyUI Manager (Recommended)

Search for **Woosh** in ComfyUI Manager and click Install.

### Manual Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/saganaki22/ComfyUI-Woosh.git
pip install -r ComfyUI-Woosh/requirements.txt
```

## Nodes

<details>
<summary><strong>Woosh Model Loader</strong> — Load generative model (Flow/DFlow/VFlow/DVFlow)</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| model_name | COMBO | Model checkpoint folder |
| model_type | COMBO | Flow, DFlow, VFlow, or DVFlow |

</details>

<details>
<summary><strong>Woosh Sampler</strong> — Generate audio from text and/or video</summary>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| gen_model | WOOSH_GEN_MODEL | | Loaded generative model |
| prompt | STRING, multiline | "" | Text description of the sound |
| steps | INT | 50 | Sampling steps (Flow: 30-100, DFlow: 4) |
| cfg | FLOAT | 4.5 | Classifier-free guidance (Flow/VFlow: 4.5, DFlow/DVFlow: 1.0) |
| seed | INT | 0 | Random seed (0 = random each time) |
| latent_frames | INT | 501 | Audio duration (100 frames ≈ 1s at 48kHz) |
| force_offload | BOOLEAN | False | Throw away model from GPU+CPU RAM after run |

**Optional Inputs:**
- `text_conditioning` — External CLAP text conditioner
- `video` — Video input for V2A mode (auto-detected)

**Outputs:**
- `video_frames` — IMAGE tensor (V2A only, for VideoCombine)
- `audio` — AUDIO tensor

</details>

<details>
<summary><strong>Woosh Video Loader</strong> — Load video file or accept image batch</summary>

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| video_path | STRING | "" | Path to video file (.mp4, .avi, etc.) |
| max_duration_s | FLOAT | 8.0 | Max video duration in seconds |
| image_batch | IMAGE | optional | Image frames as video input |

</details>

<details>
<summary><strong>Woosh TextConditioning</strong> — Load CLAP text conditioner</summary>

| Parameter | Type | Description |
|-----------|------|-------------|
| mode | COMBO | T2A (text-to-audio) or V2A (video-to-audio) |

</details>

## Model Types

| Type | Task | Steps | CFG | Description |
|------|------|-------|-----|-------------|
| **Flow** | Text-to-Audio | 50 | 4.5 | Base model, best quality |
| **DFlow** | Text-to-Audio | 4 | 1.0 | Distilled Flow, fast generation |
| **VFlow** | Video-to-Audio | 50 | 4.5 | Base video-to-audio model |
| **DVFlow** | Video-to-Audio | 4 | 1.0 | Distilled VFlow, fast video-to-audio |

> **Important:** When using the **Woosh TextConditioning** node, set `mode` to match your task:
> - **T2A** for Flow / DFlow (text-to-audio)
> - **V2A** for VFlow / DVFlow (video-to-audio)

## Model Download

Download model checkpoints from [drbaph/Woosh](https://huggingface.co/drbaph/Woosh) on HuggingFace and place each folder in `ComfyUI/models/woosh/`:

```
ComfyUI/models/woosh/
  Woosh-Flow/
    config.yaml
    weights.safetensors
  Woosh-DFlow/
    config.yaml
    weights.safetensors
  Woosh-VFlow-8s/
    config.yaml
    weights.safetensors
  Woosh-DVFlow-8s/
    config.yaml
    weights.safetensors
  Woosh-AE/
    config.yaml
    weights.safetensors
  TextConditionerA/
    config.yaml
    weights.safetensors
  TextConditionerV/
    config.yaml
    weights.safetensors
```

Each folder must contain `config.yaml` and `weights.safetensors` at the root. **Woosh-AE, TextConditionerA, and TextConditionerV are required** — all generative models reference them internally.

## VRAM Requirements

| Model | VRAM (Approx) |
|-------|---------------|
| Flow / VFlow | ~8-12 GB |
| DFlow / DVFlow | ~4-6 GB |
| With CPU offload | ~2-4 GB |

## Troubleshooting

<details>
<summary>Common issues</summary>

### "Error loading state_dict in strict mode"
Normal — some checkpoint keys don't match and non-strict loading handles this.

### RoBERTa/HuggingFace downloads every restart
The first time RoBERTa tokenizer downloads from HuggingFace, then it's cached locally in `models/woosh/hf_cache/`. Subsequent runs use the cache.

### CUDA out of memory
- Enable `force_offload` on the sampler node to fully unload after each run
- Use DFlow/DVFlow instead of Flow/VFlow (smaller model)
- Reduce `latent_frames` (501 ≈ 5s, 301 ≈ 3s)

### Model download fails (China)
Set the HuggingFace mirror before starting ComfyUI:
```bash
set HF_ENDPOINT=https://hf-mirror.com
```

### Import errors after install
Restart ComfyUI completely to reload Python modules.

</details>

## Workflow Examples

### Text-to-Audio
```
Woosh Model Loader → Woosh Sampler
```

### Video-to-Audio
```
Woosh Video Loader → Woosh Model Loader (VFlow/DVFlow) → Woosh Sampler
```

### Video-to-Audio with combined output
```
Woosh Video Loader → Woosh Model Loader (VFlow/DVFlow) → Woosh Sampler → VideoCombine
  (video_frames)  (video_frames → images)             (audio)
```

> **Tip:** Install [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) to combine video frames and audio into a single video file with the VideoCombine node.

## Credits

- **Woosh** — [SonyResearch/Woosh](https://github.com/SonyResearch/Woosh) — Sound Effect Foundation Model
- **ComfyUI Node** — [saganaki22/ComfyUI-Woosh](https://github.com/saganaki22/ComfyUI-Woosh) — This custom node

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

This custom node is released under the Apache 2.0 License. The Woosh model has its own license — see [SonyResearch/Woosh](https://github.com/SonyResearch/Woosh) for details.
