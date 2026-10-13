"""Per-model capability tables for the Vertex AI Gemini models this pack calls,
plus helpers that turn ComfyUI widget values into a request the API will accept.

Figures verified against the Gemini Enterprise Agent Platform docs (the
successor to the Vertex AI generative docs) on 2026-09-01, and re-checked
against the docs and live API calls on 2026-10-08.

This lives in its own module on purpose: it is a file upstream does not have,
so it can never conflict when merging upstream changes.
"""

from google.genai import types


# --------------------------------------------------------------------------
# Thinking
# --------------------------------------------------------------------------
# Gemini 3 and later take `thinking_level` (discrete). Gemini 2.5 and earlier
# take `thinking_budget` (token count). Sending both to a Gemini 3 model is an
# API error, and sending thinking_level to a pre-3 model is an API error too.

THINKING_LEVEL_CHOICES = ["auto", "minimal", "low", "medium", "high"]

# Text models offered by the Chat and Diarisation nodes, newest first. The 2.5
# models retire 2026-10-20 and the gemini-flash-latest / -lite-latest aliases
# 404 on Vertex, so none of them are listed.
TEXT_MODELS = [
    "gemini-4-argon",            # global only on this project; ignores temperature/top_p
    "gemini-3.8-flash",
    "gemini-3.7-flash",          # retires 2027-01-28
    "gemini-3.6-flash",          # retires 2026-11-19
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
]
DEFAULT_TEXT_MODEL = "gemini-3.8-flash"

# Supported levels per model, in ascending order. Absent = unknown, in which
# case we let the model use its own default rather than guessing.
THINKING_LEVELS = {
    "gemini-4-argon":         ("low", "medium", "high"),
    "gemini-3.8-flash":       ("low", "medium", "high"),
    "gemini-3.7-flash":       ("low", "medium", "high"),
    "gemini-3.6-flash":       ("minimal", "low", "medium", "high"),
    "gemini-3.5-flash":       ("minimal", "low", "medium", "high"),
    "gemini-3.5-flash-lite":  ("minimal", "low", "medium", "high"),
    "gemini-3.1-pro-preview": ("low", "medium", "high"),
    "gemini-3.1-flash-lite":  ("minimal", "low", "medium", "high"),
    "gemini-3-flash-preview": ("minimal", "low", "medium", "high"),
}

# thinking_budget (min, max) for the Gemini 2.5 models.
THINKING_BUDGETS = {
    "gemini-2.5-flash":      (1, 24576),
    "gemini-2.5-pro":        (128, 32768),
    "gemini-2.5-flash-lite": (512, 24576),
}

# Thinking cannot be switched off on these at all.
NO_THINKING_OFF = ("gemini-2.5-pro", "gemini-3.1-pro-preview")


def _log(msg):
    print(f"[VertexAPI] {msg}")


def _level(name):
    """types.ThinkingLevel member if the installed SDK has it, else the string."""
    return getattr(types.ThinkingLevel, name.upper(), name.upper())


# The levels in ascending order of how much the model thinks.
_SCALE = ("minimal", "low", "medium", "high")


def _nearest_level(wanted, levels):
    """Closest supported level to `wanted` on the scale, so asking for less
    thinking never silently gets you more (MINIMAL on 3.7 Flash -> LOW)."""
    i = _SCALE.index(wanted)
    return min(levels, key=lambda l: (abs(_SCALE.index(l) - i), _SCALE.index(l)))


def uses_thinking_level(model):
    """True for Gemini 3+. The floating aliases resolve to 3.x models today."""
    return not model.startswith("gemini-2.")


def build_thinking_config(model, thinking, thinking_level="auto",
                          thinking_budget=-1, include_thoughts=None):
    """Return a ThinkingConfig valid for `model`, or None to use its default."""
    extra = {} if include_thoughts is None else {"include_thoughts": include_thoughts}

    if uses_thinking_level(model):
        levels = THINKING_LEVELS.get(model)

        if not thinking:
            if levels is None:
                _log(f"{model}: unrecognised model, letting it choose its own thinking level.")
                return types.ThinkingConfig(**extra) if extra else None
            floor = levels[0]
            if model in NO_THINKING_OFF:
                _log(f"{model} cannot disable thinking; using thinking_level={floor.upper()}.")
            else:
                _log(f"{model} is a Gemini 3+ model and has no zero budget; "
                     f"using thinking_level={floor.upper()} as the closest equivalent to off.")
            return types.ThinkingConfig(thinking_level=_level(floor), **extra)

        wanted = (thinking_level or "auto").lower()
        if wanted == "auto":
            return types.ThinkingConfig(**extra) if extra else None
        if levels and wanted not in levels:
            fallback = _nearest_level(wanted, levels)
            _log(f"{model} does not support thinking_level={wanted.upper()} "
                 f"(supported: {', '.join(l.upper() for l in levels)}); using {fallback.upper()}.")
            wanted = fallback
        return types.ThinkingConfig(thinking_level=_level(wanted), **extra)

    # Gemini 2.5 and earlier.
    if thinking_level and thinking_level.lower() != "auto":
        _log(f"{model} predates thinking_level; ignoring it and using thinking_budget instead.")

    if not thinking:
        if model in NO_THINKING_OFF:
            _log(f"{model} cannot disable thinking; using the automatic budget.")
            return types.ThinkingConfig(thinking_budget=-1, **extra)
        return types.ThinkingConfig(thinking_budget=0, **extra)

    budget = thinking_budget
    lo, hi = THINKING_BUDGETS.get(model, (0, 24576))
    if budget not in (-1, 0) and not lo <= budget <= hi:
        clamped = min(max(budget, lo), hi)
        _log(f"{model} accepts a thinking_budget of {lo}-{hi}; clamping {budget} to {clamped}.")
        budget = clamped
    return types.ThinkingConfig(thinking_budget=budget, **extra)


# --------------------------------------------------------------------------
# Image generation (Nano Banana)
# --------------------------------------------------------------------------

# Full set supported by Nano Banana Pro and Nano Banana 2.
_AR_FULL = ("1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
            "9:16", "16:9", "21:9", "9:21", "1:4", "4:1", "1:8", "8:1")
# Nano Banana 2 Lite: the same minus 9:21.
_AR_LITE = tuple(a for a in _AR_FULL if a != "9:21")
# Legacy 2.5 Flash Image: the ten common ones only.
_AR_LEGACY = ("1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9")

# Per-model keys:
#   label            the name the Nano Banana v2 node shows in its model dropdown.
#   sampling         temperature / top_p / seed are accepted. When False the API
#                    rejects them outright, so they are left out of the request.
#   locations        the only locations the model is served from (None = any).
#   thinking_levels  levels the model accepts for image generation; empty = the
#                    model takes no thinking config at all.
#   image_search     Google Image Search grounding (on top of web search).
IMAGE_MODEL_CAPS = {
    "gemini-nano-banana-2.1": {
        "label": "Nano Banana 2.1",
        "aspect_ratios": _AR_FULL,
        "resolutions": ("1K", "2K", "4K"),     # no 512
        "google_search": True,
        "image_search": True,
        "max_images": 14,
        "sampling": False,
        "locations": ("global",),
        "thinking_levels": ("minimal", "medium", "high"),   # default medium
    },
    "gemini-3-pro-image": {
        "label": "Nano Banana Pro",
        "aspect_ratios": _AR_FULL,
        "resolutions": ("1K", "2K", "4K"),
        "google_search": True,
        "image_search": False,
        "max_images": 14,
        "sampling": True,
        "locations": None,
        "thinking_levels": (),                  # always thinks, no control
    },
    "gemini-3.1-flash-image": {
        "label": "Nano Banana 2",
        "aspect_ratios": _AR_FULL,
        # Note: the docs list all four, but Google currently ignores image_size
        # on this model and returns ~1K regardless (googleapis/js-genai#1461).
        "resolutions": ("512", "1K", "2K", "4K"),
        "google_search": True,
        "image_search": True,
        "max_images": 14,
        "sampling": True,
        "locations": None,
        "thinking_levels": ("minimal", "high"),             # default minimal
    },
    "gemini-3.1-flash-lite-image": {
        "label": "Nano Banana 2 Lite",
        "aspect_ratios": _AR_LITE,
        "resolutions": ("1K",),
        "google_search": False,
        "image_search": False,
        "max_images": 14,
        "sampling": True,
        "locations": None,
        "thinking_levels": (),
    },
    "gemini-2.5-flash-image": {
        "label": "Nano Banana (Gemini 2.5)",
        "aspect_ratios": _AR_LEGACY,
        "resolutions": (),          # no output resolution control at all
        "google_search": False,
        "image_search": False,
        "max_images": 3,
        "sampling": True,
        "locations": None,
        "thinking_levels": (),
    },
}

# Dropdown contents: the union, so one list serves every model. Anything the
# selected model can't do is caught at run time with a specific message.
ASPECT_RATIOS = list(_AR_FULL)
RESOLUTIONS = ["512", "1K", "2K", "4K"]
IMAGE_THINKING_CHOICES = ["auto", "minimal", "medium", "high"]
SEARCH_MODES = ["off", "web", "web + image"]


def supports_google_search(model):
    caps = IMAGE_MODEL_CAPS.get(model)
    return True if caps is None else caps["google_search"]


def supports_sampling(model):
    caps = IMAGE_MODEL_CAPS.get(model)
    return True if caps is None else caps["sampling"]


def check_location(model, location):
    caps = IMAGE_MODEL_CAPS.get(model)
    if caps and caps["locations"] and location.strip() not in caps["locations"]:
        raise ValueError(
            f"{model} is only served from: {', '.join(caps['locations'])}. "
            f"Set location to {caps['locations'][0]}."
        )


def build_search_tool(model, mode):
    """Grounding tool for the google_search widget, or None for 'off'."""
    if mode == "off":
        return None
    caps = IMAGE_MODEL_CAPS.get(model)
    if not supports_google_search(model):
        _log(f"Ignoring google_search: {model} does not support it.")
        return None
    if mode == "web + image":
        if caps is not None and not caps["image_search"]:
            raise ValueError(f"{model} does not support Google Image Search grounding.")
        return types.Tool(google_search=types.GoogleSearch(
            search_types=types.SearchTypes(
                web_search=types.WebSearch(),
                image_search=types.ImageSearch(),
            )))
    return types.Tool(google_search=types.GoogleSearch())


def build_image_thinking_config(model, thinking_level):
    """ThinkingConfig for an image model, or None to use the model's default."""
    wanted = (thinking_level or "auto").lower()
    if wanted == "auto":
        return None
    caps = IMAGE_MODEL_CAPS.get(model)
    levels = caps["thinking_levels"] if caps else None
    if caps is not None and not levels:
        _log(f"Ignoring thinking_level: {model} has no thinking control.")
        return None
    if levels and wanted not in levels:
        fallback = _nearest_level(wanted, levels)
        _log(f"{model} does not support thinking_level={wanted.upper()} "
             f"(supported: {', '.join(l.upper() for l in levels)}); using {fallback.upper()}.")
        wanted = fallback
    return types.ThinkingConfig(thinking_level=_level(wanted))


def check_image_count(model, count):
    caps = IMAGE_MODEL_CAPS.get(model)
    if caps and count > caps["max_images"]:
        raise ValueError(
            f"{model} accepts at most {caps['max_images']} input images, "
            f"but {count} are connected."
        )


def build_image_config(model, aspect_ratio, resolution):
    """Validate the widget values against `model` and return ImageConfig kwargs.
    aspect_ratio "auto" sends none, so the model follows the input image (or
    its own default when there is none)."""
    caps = IMAGE_MODEL_CAPS.get(model)
    params = {} if aspect_ratio == "auto" else {"aspect_ratio": aspect_ratio}
    if caps is None:
        return params

    if params and aspect_ratio not in caps["aspect_ratios"]:
        raise ValueError(
            f"{model} does not support the {aspect_ratio} aspect ratio. "
            f"Supported: {', '.join(caps['aspect_ratios'])}."
        )

    if caps["resolutions"]:
        if resolution not in caps["resolutions"]:
            raise ValueError(
                f"{model} does not support the {resolution} output resolution. "
                f"Supported: {', '.join(caps['resolutions'])}."
            )
        params["image_size"] = resolution
    elif resolution != "1K":
        _log(f"{model} has no output resolution control; ignoring resolution={resolution}.")
    return params


# --------------------------------------------------------------------------
# Nano Banana v2 node: what its dynamic UI shows per model
# --------------------------------------------------------------------------

LOCATIONS = [
    "global", "us-central1", "us-east1", "us-east4", "us-east5", "us-south1",
    "us-west1", "us-west2", "us-west3", "us-west4",
    "northamerica-northeast1", "northamerica-northeast2",
    "southamerica-east1", "southamerica-west1", "africa-south1",
    "europe-west1", "europe-north1", "europe-west2", "europe-west3",
    "europe-west4", "europe-west6", "europe-west8", "europe-west9",
    "europe-west12", "europe-southwest1", "europe-central2",
    "asia-east1", "asia-east2", "asia-northeast1", "asia-northeast2",
    "asia-northeast3", "asia-south1", "asia-south2", "asia-southeast1",
    "asia-southeast2", "australia-southeast1", "australia-southeast2",
    "me-central1", "me-central2", "me-west1",
]

IMAGE_MODEL_LABELS = {caps["label"]: model for model, caps in IMAGE_MODEL_CAPS.items()}


def image_model_id(label):
    """Model ID for a v2 dropdown label (an ID passes straight through)."""
    return IMAGE_MODEL_LABELS.get(label, label)


def image_ui_caps():
    """Per-label UI rules for the v2 node's frontend script. For each widget,
    `options` is the dropdown list for that model, and `disabled` names the
    widgets the model does not use, which the script greys out. The Python
    validation above stays the authority; this only shapes the UI."""
    ui = {}
    for model, caps in IMAGE_MODEL_CAPS.items():
        disabled = []
        if not caps["sampling"]:
            disabled += ["temperature", "top_p", "seed"]
        if not caps["resolutions"]:
            disabled.append("resolution")
        if not caps["thinking_levels"]:
            disabled.append("thinking_level")
        if not caps["google_search"]:
            disabled.append("google_search")
        ui[caps["label"]] = {
            "model_id": model,
            "max_images": caps["max_images"],
            "disabled": disabled,
            "options": {
                "aspect_ratio": ["auto", *caps["aspect_ratios"]],
                "resolution": list(caps["resolutions"]) or None,
                "thinking_level": ["auto", *caps["thinking_levels"]] if caps["thinking_levels"] else None,
                "google_search": (SEARCH_MODES if caps["image_search"] else SEARCH_MODES[:2])
                                 if caps["google_search"] else None,
                "location": list(caps["locations"] or LOCATIONS),
            },
        }
    return ui
