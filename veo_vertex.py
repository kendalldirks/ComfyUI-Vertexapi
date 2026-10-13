import asyncio
import io
import json
import torch
import numpy as np
import av
from PIL import Image
from google import genai
from google.genai import types
from google.oauth2 import service_account

from .vertex_async import run_blocking
from .vertex_credentials import (
    CREDENTIALS_TOOLTIP,
    default_credentials,
    list_credentials,
    resolve_credentials,
)

# Verified against the Veo 3.1 model page and live calls on 2026-10-08. The
# -preview IDs 404 on Vertex now; Veo is served from us-central1 only.
FOUR_K_MODELS = ("veo-3.1-generate-001",)
NO_REFERENCE_MODELS = ("veo-3.1-lite-generate-001",)
DURATIONS = (4, 6, 8)
MAX_REFERENCES = 3

class GoogleVeoVertexVideoGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "a cat reading a book"}),
                "credentials": (list_credentials(), {"default": default_credentials(),
                                                     "tooltip": CREDENTIALS_TOOLTIP}),
                "location": (["us-central1"], {"default": "us-central1",
                                               "tooltip": "Veo is served from us-central1 only."}),
                "model": ([
                    "veo-3.1-generate-001",
                    "veo-3.1-fast-generate-001",
                    "veo-3.1-lite-generate-001"
                ], {"default": "veo-3.1-generate-001"}),
                "resolution": (["720p", "1080p", "4k"], {"default": "720p",
                                                         "tooltip": "4k is veo-3.1-generate-001 only."}),
                "aspect_ratio": (["16:9", "9:16"], {"default": "16:9"}),
                "duration_seconds": ("INT", {"default": 4, "min": 4, "max": 8, "step": 2,
                                             "tooltip": "4, 6 or 8 seconds."}),
                "seed": ("INT", {"default": 69, "min": 1, "max": 2147483646, "step": 1}),
                "generate_audio": ("BOOLEAN", {"default": False}),
                "fps": (["24"], {"default": "24"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "number_of_videos": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1,
                                             "tooltip": "Videos per request. Each comes out as its own item "
                                                        "on the frames and audio lists."}),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
                **{f"reference_{i}": ("IMAGE", {"tooltip": "Asset reference (a person, character or product) to keep "
                                                            "consistent in the video. Not on veo-3.1-lite; "
                                                            "veo-3.1-generate-001 needs 8 seconds with references."})
                   for i in range(1, MAX_REFERENCES + 1)},
                "project_id": ("STRING", {"forceInput": True, "tooltip": "Overrides the credentials file."}),
                "service_account": ("STRING", {"forceInput": True, "tooltip": "Overrides the credentials file."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("frames", "audio")
    FUNCTION = "generate_video"
    CATEGORY = "video/generation"
    OUTPUT_IS_LIST = (True, True)

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

    async def generate_video(self, prompt, credentials, location, model, resolution, aspect_ratio,
                       duration_seconds, seed, generate_audio, fps, negative_prompt=None,
                       number_of_videos=1, first_frame=None, last_frame=None,
                       project_id=None, service_account=None, **kwargs):

        references = [kwargs[f"reference_{i}"] for i in range(1, MAX_REFERENCES + 1)
                      if kwargs.get(f"reference_{i}") is not None]
        if duration_seconds not in DURATIONS:
            raise ValueError(f"Veo supports durations of 4, 6 or 8 seconds, not {duration_seconds}.")
        if resolution == "4k" and model not in FOUR_K_MODELS:
            raise ValueError(f"{model} does not support 4k. Use 720p/1080p, or veo-3.1-generate-001 for 4k.")
        if references and model in NO_REFERENCE_MODELS:
            raise ValueError(f"{model} does not support reference images.")
        if references and model == "veo-3.1-generate-001" and duration_seconds != 8:
            raise ValueError("veo-3.1-generate-001 only supports 8-second videos with reference images.")

        project_id, service_account = resolve_credentials(credentials, project_id, service_account)
        client = self.setup_client(service_account, project_id, location)

        def tensor_to_bytes(t):
            if t.dim() == 4: t = t[0]
            arr = (t.cpu().numpy() * 255).astype(np.uint8)
            b = io.BytesIO()
            Image.fromarray(arr).save(b, format="PNG")
            return b.getvalue()

        config = types.GenerateVideosConfig(
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            duration_seconds=duration_seconds,
            generate_audio=generate_audio,
            fps=int(fps),
            seed=seed if seed != -1 else None,
            number_of_videos=number_of_videos,
            negative_prompt=negative_prompt.strip() if negative_prompt and negative_prompt.strip() else None,
            reference_images=[
                types.VideoGenerationReferenceImage(
                    image=types.Image(image_bytes=tensor_to_bytes(r), mime_type="image/png"),
                    reference_type=types.VideoGenerationReferenceType.ASSET,
                ) for r in references
            ] or None,
        )

        gen_kwargs = {"model": model, "prompt": prompt, "config": config}

        if first_frame is not None:
            gen_kwargs["image"] = types.Image(image_bytes=tensor_to_bytes(first_frame), mime_type="image/png")
        if last_frame is not None:
            setattr(config, 'last_frame', types.Image(image_bytes=tensor_to_bytes(last_frame), mime_type="image/png"))

        op = await run_blocking(client.models.generate_videos, **gen_kwargs)
        print(f"Veo Operation: {op.name}")

        while not op.done:
            # asyncio.sleep, not time.sleep: a Veo job runs for minutes and must
            # not hold the event loop hostage while it polls.
            await asyncio.sleep(5)
            op = await run_blocking(client.operations.get, op)

        if op.error: raise Exception(f"Veo Error: {op.error}")
        if not op.result or not op.result.generated_videos:
            reasons = getattr(op.result, "rai_media_filtered_reasons", None) if op.result else None
            raise Exception("No videos generated" + (f": filtered by safety ({'; '.join(reasons)})" if reasons else ""))

        filtered = getattr(op.result, "rai_media_filtered_count", None)
        if filtered:
            print(f"[VertexAPI] Veo: {filtered} of {number_of_videos} videos were removed by safety filters.")

        all_frames, all_audio = [], []
        for generated in op.result.generated_videos:
            frames, audio = self._decode(generated.video.video_bytes, generate_audio)
            all_frames.append(frames)
            all_audio.append(audio)

        return (all_frames, all_audio)

    @staticmethod
    def _decode(data, generate_audio):
        video_bytes = io.BytesIO(data)

        container = av.open(video_bytes)
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_rgb().to_ndarray().astype(np.float32) / 255.0
            frames.append(torch.from_numpy(img).unsqueeze(0))
        container.close()

        audio = None
        if generate_audio:
            video_bytes.seek(0)
            container = av.open(video_bytes)
            if container.streams.audio:
                audio_data = [f.to_ndarray() for f in container.decode(audio=0)]
                if audio_data:
                    waveform = torch.from_numpy(np.concatenate(audio_data, axis=1)).float()
                    if audio_data[0].dtype == np.int16: waveform /= 32768.0
                    elif audio_data[0].dtype == np.int32: waveform /= 2147483648.0
                    audio = {"waveform": waveform.unsqueeze(0), "sample_rate": container.streams.audio[0].rate}
            container.close()

        if not frames: raise Exception("Failed to decode video frames")

        return torch.cat(frames, dim=0), audio


NODE_CLASS_MAPPINGS = {"GoogleVeoVertexVideoGenerator": GoogleVeoVertexVideoGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"GoogleVeoVertexVideoGenerator": "Google Veo (Vertex AI)"}