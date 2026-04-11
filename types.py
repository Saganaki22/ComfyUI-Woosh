"""ComfyUI custom types for type-safe wiring between Woosh nodes."""

# Custom Woosh types
GEN_MODEL = "WOOSH_GEN_MODEL"       # Flow, DFlow, VFlow, or DVFlow
TEXT_COND = "WOOSH_TEXT_COND"
VIDEO = "WOOSH_VIDEO"

# Native ComfyUI type — output this so built-in preview/save nodes work
AUDIO = "AUDIO"
