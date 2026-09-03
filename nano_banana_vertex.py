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
    RESOLUTIONS,
    build_image_config,
    check_image_count,
    supports_google_search,
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
                    "gemini-3-pro-image",
                    "gemini-3.1-flash-image",
                    "gemini-3.1-flash-lite-image",
                    "gemini-2.5-flash-image"
                ], {"default": "gemini-3-pro-image"}),
                "aspect_ratio": (ASPECT_RATIOS, {"default": "1:1"}),
                "resolution": (RESOLUTIONS, {"default": "1K"}),
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "google_search": ("BOOLEAN", {"default": False}),
                "retry_on_empty": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_instruction": ("STRING", {"multiline": True, "default": ""}),
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
                       prompt="", system_instruction="", **kwargs):

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

        tools = None
        if google_search:
            if supports_google_search(model):
                tools = [types.Tool(googleSearch=types.GoogleSearch())]
            else:
                print(f"[VertexAPI] Ignoring google_search: {model} does not support it.")

        img_config_params = build_image_config(model, aspect_ratio, resolution)

        def make_config(seed_value):
            return types.GenerateContentConfig(
                temperature=temperature,
                seed=seed_value,
                top_p=top_p,
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(**img_config_params),
                system_instruction=system_instruction.strip() if system_instruction.strip() else None,
                tools=tools
            )

        contents = [types.Content(role="user", parts=parts)]
        attempts = EMPTY_RETRY_ATTEMPTS if retry_on_empty else 1

        for attempt in range(1, attempts + 1):
            # A repeat of the same seed re-rolls to the same empty result, so
            # step it. seed -1 already means "let the API pick", so leave it.
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
                nxt = "the same random seed" if seed < 0 else f"seed {seed + attempt}"
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