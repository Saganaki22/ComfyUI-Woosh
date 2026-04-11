"""Dynamic VRAM management for Woosh models — extends ComfyUI's ModelPatcher.

Models live on CPU (offload_device) by default. ComfyUI's memory manager
calls model_patches_to() to move to GPU, and detach() to move back.
"""

import gc
import torch
import comfy.model_management as mm
from comfy.model_patcher import ModelPatcher


class WooshModelPatcher(ModelPatcher):
    """ModelPatcher subclass for Woosh models.

    Handles GPU<->CPU movement with proper VRAM tracking.
    ComfyUI's memory manager calls model_patches_to() to load to GPU
    and detach() to unload.
    """

    def __init__(self, model, vram_size_gb: float = 4.0, reload_fn=None):
        load_device = mm.get_torch_device()
        offload_device = mm.intermediate_device()
        size = int(vram_size_gb * (1024 ** 3))
        super().__init__(model, load_device, offload_device, size=size)
        self._reload_fn = reload_fn
        # Start on CPU
        self.model.to(offload_device)

    def model_patches_to(self, device):
        """Move model to target device (called by ComfyUI to load to GPU)."""
        if self.model is None and self._reload_fn is not None:
            self.model = self._reload_fn()
        if self.model is not None:
            self.model.to(device)

    def detach(self, unpatch_weights=True, **kwargs):
        """Move model off GPU (called by ComfyUI to free VRAM)."""
        if self.model is not None:
            self.model.to(self.offload_device)
            mm.soft_empty_cache()

    def model_size(self, device=None):
        """Report total model size for VRAM budgeting."""
        return self.size

    def loaded_size(self):
        """Report currently loaded memory. 0 if on CPU."""
        if self.current_loaded_device() == self.offload_device:
            return 0
        return self.size

    def model_mmap_residency(self, free=False):
        return 0, 0  # Woosh models don't use mmap

    def current_loaded_device(self):
        try:
            if self.model is None:
                return self.offload_device
            return next(self.model.parameters()).device
        except StopIteration:
            return self.offload_device

    def partially_unload(self, device, memory_to_free):
        """ComfyUI asks to free some memory. We unload everything."""
        self.detach()
        return self.size

    def partially_load(self, device, memory_to_free, force_patch_weights=False):
        """ComfyUI asks to load some memory. We load everything."""
        self.model_patches_to(device)
        return self.size

    def force_unload(self):
        """Throw away model from GPU + CPU. Reloads from disk next use."""
        if self.model is not None:
            self.model.to(self.offload_device)
        self.model = None
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        mm.soft_empty_cache()
