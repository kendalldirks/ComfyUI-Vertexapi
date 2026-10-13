from . import (
    gemini_node_vertex,
    # nano_banana_vertex.py is the previous Nano Banana node, kept as a backup.
    # To go back to it, swap nano_banana_vertex_v2 for nano_banana_vertex here
    # and in the list below.
    nano_banana_vertex_v2,
    gemini_tts_vertex,
    veo_vertex,
    gemini_diarisation_vertex,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Frontend scripts (the Nano Banana v2 dynamic UI).
WEB_DIRECTORY = "./web"

modules = [
    gemini_node_vertex,
    nano_banana_vertex_v2,
    gemini_tts_vertex,
    veo_vertex,
    gemini_diarisation_vertex,
]

for module in modules:
    NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {}))

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
