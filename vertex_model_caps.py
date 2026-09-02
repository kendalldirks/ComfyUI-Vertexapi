"""Per-model capability tables for the Vertex AI Gemini models this pack calls,
plus helpers that turn ComfyUI widget values into a request the API will accept.

Figures verified against the Gemini Enterprise Agent Platform docs (the
successor to the Vertex AI generative docs) on 2026-09-01.

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

# Supported levels per model, in ascending order. Absent = unknown, in which
# case we let the model use its own default rather than guessing.
THINKING_LEVELS = {
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
                _log(f"{model} is a Gemini 3 model and has no zero budget; "
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

IMAGE_MODEL_CAPS = {
    "gemini-3-pro-image": {
        "aspect_ratios": _AR_FULL,
        "resolutions": ("1K", "2K", "4K"),
        "google_search": True,
        "max_images": 14,
    },
    "gemini-3.1-flash-image": {
        "aspect_ratios": _AR_FULL,
        # Note: the docs list all four, but Google currently ignores image_size
        # on this model and returns ~1K regardless (googleapis/js-genai#1461).
        "resolutions": ("512", "1K", "2K", "4K"),
        "google_search": True,
        "max_images": 14,
    },
    "gemini-3.1-flash-lite-image": {
        "aspect_ratios": _AR_LITE,
        "resolutions": ("1K",),
        "google_search": False,
        "max_images": 14,
    },
    "gemini-2.5-flash-image": {
        "aspect_ratios": _AR_LEGACY,
        "resolutions": (),          # no output resolution control at all
        "google_search": False,
        "max_images": 3,
    },
}

# Dropdown contents: the union, so one list serves every model. Anything the
# selected model can't do is caught at run time with a specific message.
ASPECT_RATIOS = list(_AR_FULL)
RESOLUTIONS = ["512", "1K", "2K", "4K"]


def supports_google_search(model):
    caps = IMAGE_MODEL_CAPS.get(model)
    return True if caps is None else caps["google_search"]


def check_image_count(model, count):
    caps = IMAGE_MODEL_CAPS.get(model)
    if caps and count > caps["max_images"]:
        raise ValueError(
            f"{model} accepts at most {caps['max_images']} input images, "
            f"but {count} are connected."
        )


def build_image_config(model, aspect_ratio, resolution):
    """Validate the widget values against `model` and return ImageConfig kwargs."""
    caps = IMAGE_MODEL_CAPS.get(model)
    if caps is None:
        return {"aspect_ratio": aspect_ratio}

    if aspect_ratio not in caps["aspect_ratios"]:
        raise ValueError(
            f"{model} does not support the {aspect_ratio} aspect ratio. "
            f"Supported: {', '.join(caps['aspect_ratios'])}."
        )

    params = {"aspect_ratio": aspect_ratio}
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
