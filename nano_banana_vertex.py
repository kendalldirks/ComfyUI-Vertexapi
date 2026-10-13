import asyncio
import io
import json
import torch
import numpy as np
from PIL import Image
from google import genai
from google.genai import types
from google.oauth2 import service_account

from .vertex_async import run_blocking
from .vertex_model_caps import (
    ASPECT_RATIOS,
    IMAGE_THINKING_CHOICES,
    RESOLUTIONS,
    SEARCH_MODES,
    build_image_config,
    build_image_thinking_config,
    build_search_tool,
    check_image_count,
    check_location,
    supports_sampling,
)
from .vertex_response import EmptyResponseError, extract_image_bytes

# Total attempts when retry_on_empty is on, and the pause between them.
EMPTY_RETRY_ATTEMPTS = 3
EMPTY_RETRY_DELAY = 1.0

MAX_IMAGE_SLOTS = 14

class NanoBananaVertexNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ComfyUI draws every "required" widget before every "optional"
                # one, so widget order is just declaration order. prompt and
                # system_instruction live in "optional" at the bottom so the two
                # text boxes sit together.
                "project_id": ("STRING", {"multiline": False, "default": ""}),
                "location": ([
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
                    "me-central1", "me-central2", "me-west1"
                ], {"default": "us-central1"}),
                "service_account": ("STRING", {"multiline": True, "default": ""}),
                "model": ([
                    "gemini-nano-banana-2.1",
                    "gemini-3-pro-image",
                    "gemini-3.1-flash-image",
                    "gemini-3.1-flash-lite-image",
                    "gemini-2.5-flash-image"
                ], {"default": "gemini-nano-banana-2.1"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "1:1"}),
                "resolution": (RESOLUTIONS, {"default": "1K"}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01,
                                          "tooltip": "Ignored by gemini-nano-banana-2.1 (the API rejects it)."}),
                "top_p": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01,
                                    "tooltip": "Ignored by gemini-nano-banana-2.1 (the API rejects it)."}),
                "google_search": (SEARCH_MODES, {"default": "off",
                                                 "tooltip": "Grounding. 'web + image' adds Google Image Search "
                                                            "(Nano Banana 2.1 and 2 only)."}),
                "retry_on_empty": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1,
                                 "tooltip": "gemini-nano-banana-2.1 does not accept a seed: its output is "
                                            "random every run. Here the seed only forces a re-run."}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_instruction": ("STRING", {"multiline": True, "default": ""}),
                "thinking_level": (IMAGE_THINKING_CHOICES, {"default": "auto",
                                                            "tooltip": "auto leaves the model on its own default. "
                                                                       "A level the model lacks falls back to the nearest."}),
                **{f"image_{i}": ("IMAGE",) for i in range(1, MAX_IMAGE_SLOTS + 1)},
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "image/generation"

    def setup_client(self, service_account_json, project_id, location):
        if not service_account_json.strip():
            raise ValueError("Service account JSON content is required.")
        if not project_id.strip():
            raise ValueError("Project ID is required.")

        try:
            sa_info = json.loads(service_account_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON content: {str(e)}")

        credentials = service_account.Credentials.from_service_account_info(
            sa_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        return genai.Client(
            vertexai=True,
            project=project_id.strip(),
            location=location.strip(),
            credentials=credentials,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=10, jitter=10)
            )
        )

    def _convert_tensor_to_bytes(self, tensor):
        if tensor.dim() == 4:
            tensor = tensor[0]
        arr = (tensor.cpu().numpy() * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format='PNG')
        return buf.getvalue()

    async def generate(self, project_id, location, service_account, model, aspect_ratio, resolution,
                       temperature, top_p, google_search, retry_on_empty, seed,
                       prompt="", system_instruction="", thinking_level="auto", **kwargs):

        check_location(model, location)
        client = self.setup_client(service_account, project_id, location)

        parts = []
        image_count = 0
        for i in range(1, MAX_IMAGE_SLOTS + 1):
            img = kwargs.get(f"image_{i}")
            if img is not None:
                image_count += 1
                parts.append(types.Part.from_bytes(mime_type="image/png", data=self._convert_tensor_to_bytes(img)))
        check_image_count(model, image_count)

        if prompt.strip():
            parts.append(types.Part.from_text(text=prompt))

        if not parts:
            raise ValueError("At least one image or prompt must be provided.")

        search_tool = build_search_tool(model, google_search)
        tools = [search_tool] if search_tool else None

        img_config_params = build_image_config(model, aspect_ratio, resolution)
        thinking_config = build_image_thinking_config(model, thinking_level)
        sampling = supports_sampling(model)
        if not sampling:
            print(f"[VertexAPI] {model} rejects temperature/top_p/seed; not sending them.")

        def make_config(seed_value):
            sampling_params = {"temperature": temperature, "seed": seed_value, "top_p": top_p} if sampling else {}
            return types.GenerateContentConfig(
                **sampling_params,
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(**img_config_params),
                system_instruction=system_instruction.strip() if system_instruction.strip() else None,
                thinking_config=thinking_config,
                tools=tools
            )

        contents = [types.Content(role="user", parts=parts)]
        attempts = EMPTY_RETRY_ATTEMPTS if retry_on_empty else 1

        for attempt in range(1, attempts + 1):
            # A repeat of the same seed re-rolls to the same empty result, so
            # step it. seed -1 already means "let the API pick", so leave it.
            # (Models that take no seed just repeat; every call is random anyway.)
            attempt_seed = seed if seed < 0 else seed + attempt - 1

            try:
                response = await run_blocking(
                    client.models.generate_content,
                    model=model,
                    contents=contents,
                    config=make_config(attempt_seed),
                )
            except Exception as e:
                raise RuntimeError(f"Gemini API Error: {str(e)}")

            # Raises with the actual reason (safety stop, blocked prompt, budget
            # spent on thinking, text-only reply) rather than a bare TypeError.
            try:
                img_data = extract_image_bytes(response, model)
                break
            except EmptyResponseError as e:
                last = attempt == attempts
                if last or not e.retryable:
                    if not e.retryable and not last:
                        print(f"[VertexAPI] Not retrying: {e.reason} will repeat for this request.")
                    raise
                nxt = ("a fresh random roll" if not sampling else
                       "the same random seed" if seed < 0 else f"seed {seed + attempt}")
                print(f"[VertexAPI] Empty response on attempt {attempt}/{attempts} "
                      f"({e.reason or 'no reason given'}); retrying with {nxt}.")
                await asyncio.sleep(EMPTY_RETRY_DELAY)

        result_pil = Image.open(io.BytesIO(img_data)).convert("RGB")
        return (torch.from_numpy(np.array(result_pil).astype(np.float32) / 255.0).unsqueeze(0),)

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed


NODE_CLASS_MAPPINGS = {"NanoBananaVertexNode": NanoBananaVertexNode}
NODE_DISPLAY_NAME_MAPPINGS = {"NanoBananaVertexNode": "Nano Banana (Vertex AI)"}