"""Nano Banana v2: the Vertex Nano Banana node rebuilt on ComfyUI's V3 schema.

- Image inputs grow as you connect them (Autogrow), like the Comfy API nodes.
- web/js/nano_banana_v2.js narrows each dropdown to what the selected model
  supports and greys out the widgets it does not use. It reads the rules from
  vertex_model_caps.image_ui_caps(), sent along with the model input, so the
  UI and the run-time checks below come from the same table.

nano_banana_vertex.py is the previous node, kept on disk as a fallback; to go
back, import it in __init__.py instead of this module.
"""

import asyncio
import io
import json

import numpy as np
import torch
from PIL import Image
from google import genai
from google.genai import types
from google.oauth2 import service_account
from comfy_api.latest import io as cio

from .vertex_async import run_blocking
from .vertex_credentials import (
    CREDENTIALS_TOOLTIP,
    default_credentials,
    list_credentials,
    resolve_credentials,
)
from .vertex_model_caps import (
    IMAGE_MODEL_CAPS,
    IMAGE_THINKING_CHOICES,
    LOCATIONS,
    RESOLUTIONS,
    SEARCH_MODES,
    ASPECT_RATIOS,
    build_image_config,
    build_image_thinking_config,
    build_search_tool,
    check_image_count,
    check_location,
    image_model_id,
    image_ui_caps,
    supports_sampling,
)
from .vertex_response import EmptyResponseError, extract_image_bytes

# Total attempts when retry_on_empty is on, and the pause between them.
EMPTY_RETRY_ATTEMPTS = 3
EMPTY_RETRY_DELAY = 1.0

MAX_IMAGE_SLOTS = 14
MAX_BATCH = 4
SEED_MAX = 2147483646

# Frontend event for a batch where some generations failed (shown as a toast).
PARTIAL_FAILURE_EVENT = "vertexapi.partial_failure"

MODEL_LABELS = [caps["label"] for caps in IMAGE_MODEL_CAPS.values()]


def setup_client(service_account_json, project_id, location):
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


def image_parts(images):
    """PNG parts for every frame of every connected slot, in slot order."""
    parts = []
    for name in sorted(images, key=lambda n: int(n.rsplit("_", 1)[1])):
        tensor = images[name]
        if tensor is None:
            continue
        batch = tensor if tensor.dim() == 4 else tensor.unsqueeze(0)
        for frame in batch:
            arr = (frame.cpu().numpy() * 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="PNG")
            parts.append(types.Part.from_bytes(mime_type="image/png", data=buf.getvalue()))
    return parts


class NanoBananaVertexV2(cio.ComfyNode):

    @classmethod
    def define_schema(cls):
        return cio.Schema(
            node_id="NanoBananaVertexV2",
            display_name="Nano Banana (Vertex AI)",
            category="image/generation",
            description="Generate or edit images with the Gemini image models on Vertex AI. "
                        "Options follow the selected model; settings it does not use are greyed out.",
            inputs=[
                # Fixed sockets first, the growing image list last: ComfyUI restores
                # links by slot position when a node is pasted or loaded, so anything
                # after the images would collide with image slots that grow later.
                cio.String.Input("project_id", force_input=True, optional=True,
                                 tooltip="Overrides the credentials file."),
                cio.String.Input("service_account", force_input=True, optional=True,
                                 tooltip="Service account JSON. Overrides the credentials file."),
                cio.Autogrow.Input(
                    "images",
                    optional=True,
                    template=cio.Autogrow.TemplateNames(
                        cio.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, MAX_IMAGE_SLOTS + 1)],
                        min=0,
                    ),
                    tooltip="Reference images. A new slot appears as you connect one. "
                            "Up to 14 images (3 on Nano Banana (Gemini 2.5)); every frame of a batch counts.",
                ),
                # Widget order is the order below (the seed's control_after_generate
                # follows it). All widgets are required so none get pushed to the end.
                cio.Combo.Input("credentials", options=list_credentials(), default=default_credentials(),
                                tooltip=CREDENTIALS_TOOLTIP),
                cio.Combo.Input("location", options=LOCATIONS, default="global",
                                tooltip="Nano Banana 2.1 is served from global only."),
                cio.Combo.Input("model", options=MODEL_LABELS, default=MODEL_LABELS[0],
                                extra_dict={"vertex_caps": image_ui_caps()}),
                cio.Combo.Input("aspect_ratio", options=["auto", *ASPECT_RATIOS], default="auto",
                                tooltip="auto follows the input image, or the model's default with no image."),
                cio.Combo.Input("resolution", options=RESOLUTIONS, default="1K"),
                cio.Int.Input("batch_size", default=1, min=1, max=MAX_BATCH,
                              tooltip="Generations run at the same time, returned as one image batch. "
                                      "If some fail, the rest still return and a warning says why."),
                cio.Float.Input("temperature", default=0.5, min=0.0, max=2.0, step=0.01),
                cio.Float.Input("top_p", default=0.85, min=0.0, max=1.0, step=0.01),
                cio.String.Input("prompt", multiline=True, default=""),
                cio.String.Input("system_instruction", multiline=True, default=""),
                cio.Int.Input("seed", default=69, min=0, max=2147483646, control_after_generate=True,
                              tooltip="Not used by Nano Banana 2.1, whose output is random every run."),
                cio.Combo.Input("thinking_level", options=IMAGE_THINKING_CHOICES, default="auto",
                                tooltip="auto leaves the model on its own default."),
                cio.Combo.Input("google_search", options=SEARCH_MODES, default="off",
                                tooltip="Grounding. 'web + image' adds Google Image Search."),
                cio.Boolean.Input("retry_on_empty", default=True,
                                  tooltip="When the API returns no image (a safety trip, a recitation stop), "
                                          "try up to 3 times. Stops that would repeat are not retried."),
            ],
            outputs=[cio.Image.Output(display_name="image")],
            hidden=[cio.Hidden.unique_id],
        )

    @classmethod
    async def execute(cls, credentials, model, location, prompt, aspect_ratio, resolution,
                      thinking_level, google_search, batch_size, seed, retry_on_empty, temperature, top_p,
                      images=None, system_instruction="", project_id=None, service_account=None) -> cio.NodeOutput:
        label, model = model, image_model_id(model)

        check_location(model, location)
        project_id, service_account = resolve_credentials(credentials, project_id, service_account)

        parts = image_parts(images or {})
        check_image_count(model, len(parts))
        if prompt.strip():
            parts.append(types.Part.from_text(text=prompt))
        if not parts:
            raise ValueError("Connect at least one image or write a prompt.")

        search_tool = build_search_tool(model, google_search)
        tools = [search_tool] if search_tool else None
        img_config_params = build_image_config(model, aspect_ratio, resolution)
        thinking_config = build_image_thinking_config(model, thinking_level)
        sampling = supports_sampling(model)

        def make_config(seed_value):
            sampling_params = {"temperature": temperature, "seed": seed_value, "top_p": top_p} if sampling else {}
            return types.GenerateContentConfig(
                **sampling_params,
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(**img_config_params),
                system_instruction=system_instruction.strip() or None,
                thinking_config=thinking_config,
                tools=tools,
            )

        client = setup_client(service_account, project_id, location)
        contents = [types.Content(role="user", parts=parts)]
        attempts = EMPTY_RETRY_ATTEMPTS if retry_on_empty else 1
        print(f"[VertexAPI] {label} ({model}) at {location.strip()}, "
              f"{len(parts) - bool(prompt.strip())} input image(s), batch of {batch_size}.")

        async def generate_one(index):
            """Image bytes for batch item `index`, with its own empty-response retries.
            Each item starts `attempts` seeds apart so items and retries never share one."""
            tag = f"[VertexAPI] image {index + 1}/{batch_size}:"
            for attempt in range(1, attempts + 1):
                attempt_seed = (seed + index * attempts + attempt - 1) % (SEED_MAX + 1)
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
                    return extract_image_bytes(response, model)
                except EmptyResponseError as e:
                    last = attempt == attempts
                    if last or not e.retryable:
                        if not e.retryable and not last:
                            print(f"{tag} not retrying: {e.reason} will repeat for this request.")
                        raise
                    nxt = f"seed {attempt_seed + 1}" if sampling else "a fresh random roll"
                    print(f"{tag} empty response on attempt {attempt}/{attempts} "
                          f"({e.reason or 'no reason given'}); retrying with {nxt}.")
                    await asyncio.sleep(EMPTY_RETRY_DELAY)

        results = await asyncio.gather(*(generate_one(i) for i in range(batch_size)), return_exceptions=True)

        tensors, errors = [], []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                errors.append(f"image {i + 1}/{batch_size}: {result}")
                continue
            pil = Image.open(io.BytesIO(result)).convert("RGB")
            tensor = torch.from_numpy(np.array(pil).astype(np.float32) / 255.0)
            # An image batch must share one size. Same-settings requests come back
            # the same size, but if one doesn't it is reported, not resized.
            if tensors and tensor.shape != tensors[0].shape:
                errors.append(f"image {i + 1}/{batch_size}: came back {tensor.shape[1]}x{tensor.shape[0]}, "
                              f"not {tensors[0].shape[1]}x{tensors[0].shape[0]} like the rest of the batch, "
                              f"so it can't be stacked with them")
                continue
            tensors.append(tensor)

        if not tensors:
            raise RuntimeError("Every generation failed:\n" + "\n".join(errors))

        error_text = "\n".join(errors)
        if errors:
            print(f"[VertexAPI] {len(tensors)} of {batch_size} images returned; {len(errors)} failed:\n{error_text}")
            cls._notify_partial_failure(len(tensors), batch_size, error_text)

        return cio.NodeOutput(torch.stack(tensors))

    @classmethod
    def _notify_partial_failure(cls, succeeded, total, error_text):
        """Toast in the browser (web/js/nano_banana_v2.js listens for the event)."""
        try:
            from server import PromptServer
            PromptServer.instance.send_sync(PARTIAL_FAILURE_EVENT, {
                "node": cls.hidden.unique_id if cls.hidden else None,
                "summary": f"Nano Banana: {total - succeeded} of {total} generations failed",
                "detail": error_text,
            })
        except Exception as e:  # never let the notification sink a result
            print(f"[VertexAPI] Could not send the partial-failure notice to the browser: {e}")

NODE_CLASS_MAPPINGS = {"NanoBananaVertexV2": NanoBananaVertexV2}
NODE_DISPLAY_NAME_MAPPINGS = {"NanoBananaVertexV2": "Nano Banana (Vertex AI)"}
