"""Woosh video loader — extract frames for VFlow/DVFlow models."""

import torch
import numpy as np
from woosh.utils.videoio import extract_video_frames

import comfy.model_management as mm

from ..woosh_types import VIDEO


class WooshLoadVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "video_path": ("STRING", {"default": "", "tooltip": "Path to video file (.mp4, .avi, etc.)"}),
                "max_duration_s": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 30.0, "step": 0.5, "tooltip": "Max video duration in seconds (VFlow-8s limit is 8)"}),
                "image_batch": ("IMAGE",),
            }
        }

    RETURN_TYPES = (VIDEO,)
    RETURN_NAMES = ("video",)
    FUNCTION = "load"
    CATEGORY = "Woosh/Video"
    DESCRIPTION = "Load video file or accept image batch for video-to-audio generation"

    def load(self, video_path="", max_duration_s=8.0, image_batch=None):
        if image_batch is not None:
            # ComfyUI IMAGE is [B, H, W, C] float32 in [0,1]
            # Convert to (T, H, W, C) uint8 for SynchformerProcessor
            frames = (image_batch.cpu().numpy() * 255).astype(np.uint8)
            frames = torch.from_numpy(frames)
            return ({"frames": frames, "rate": 24},)

        if not video_path:
            raise ValueError("Must provide either video_path or image_batch")

        frames, rate, pts = extract_video_frames(
            video_path,
            start_time=0,
            end_time=max_duration_s,
        )

        return ({"frames": frames, "rate": rate},)


NODE_CLASS_MAPPINGS_VIDEO = {
    "WooshLoadVideo": WooshLoadVideo,
}

NODE_DISPLAY_MAPPINGS_VIDEO = {
    "WooshLoadVideo": "Woosh Video Loader",
}
