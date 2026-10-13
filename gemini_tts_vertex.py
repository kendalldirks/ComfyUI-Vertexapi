import io
import json
import re
import wave
import torch
from google.genai import Client, types
from google.oauth2 import service_account

from .vertex_async import run_blocking
from .vertex_credentials import (
    CREDENTIALS_TOOLTIP,
    default_credentials,
    list_credentials,
    resolve_credentials,
)

VOICES = ["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Achernar", "Laomedeia", "Rasalgethi", "Algenib", "Achird", "Pulcherrima", "Gacrux", "Schedar", "Alnilam", "Sulafat", "Sadaltager", "Sadachbia", "Vindemiatrix", "Zubenelgenubi"]

# The 3.8 TTS models are served from `global` only (404 anywhere else).
GLOBAL_ONLY_MODELS = ("gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts")


def decode_audio(data, mime_type):
    """(int16 samples, sample_rate) from a TTS reply. The 3.8 models return a
    WAV file (header included); older models return headerless 16-bit PCM with
    the rate in the mime type, e.g. audio/L16;codec=pcm;rate=24000."""
    if data[:4] == b"RIFF":
        with wave.open(io.BytesIO(data)) as w:
            if w.getsampwidth() != 2:
                raise ValueError(f"Unexpected WAV sample width {w.getsampwidth() * 8}-bit; expected 16-bit.")
            frames = w.readframes(w.getnframes())
            rate, channels = w.getframerate(), w.getnchannels()
        samples = torch.frombuffer(bytearray(frames), dtype=torch.int16)
        if channels > 1:
            samples = samples.view(-1, channels).float().mean(dim=1).to(torch.int16)
        return samples, rate
    m = re.search(r"rate=(\d+)", mime_type or "")
    return torch.frombuffer(bytearray(data), dtype=torch.int16), int(m.group(1)) if m else 24000


class GeminiTTSVertexNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "",
                                    "tooltip": "Inline tags such as <short pause>, <sigh> or <cough> work in the text. "
                                               "With two_speakers on, write each line as 'Name: line' using the two speaker names."}),
                "credentials": (list_credentials(), {"default": default_credentials(),
                                                     "tooltip": CREDENTIALS_TOOLTIP}),
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
                ], {"default": "global",
                    "tooltip": "The 3.8 TTS models are served from global only."}),
                "model": ([
                    "gemini-3.8-flash-tts",
                    "gemini-3.8-flash-lite-tts",
                    "gemini-3.1-flash-tts-preview",
                    "gemini-2.5-pro-tts",
                    "gemini-2.5-flash-tts"
                ], {"default": "gemini-3.8-flash-tts"}),
                "voice_id": (VOICES, {"tooltip": "The voice, or speaker 1's voice when two_speakers is on."}),
                "seed": ("INT", {"default": 69, "min": -1, "max": 2147483646, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "optional": {
                "style": (["None", "Vocal Smile", "Newscaster", "Whisper", "Empathetic", "Promo/Hype", "Deadpan"], {"default": "None"}),
                "pace": (["None", "Natural", "Rapid Fire", "The Drift", "Staccato"], {"default": "None"}),
                "accent": (["None", "Neutral", "American (Gen)", "American (Valley)", "American (South)", "British (RP)", "Transatlantic", "Australian"], {"default": "None"}),
                "audio_profile": ("STRING", {"multiline": True, "default": ""}),
                "scene": ("STRING", {"multiline": True, "default": ""}),
                "two_speakers": ("BOOLEAN", {"default": False,
                                             "tooltip": "Dialogue between exactly two voices. Write the text as "
                                                        "'Name: line' per line, using the speaker names below."}),
                "speaker_1_name": ("STRING", {"default": "Speaker1"}),
                "speaker_2_name": ("STRING", {"default": "Speaker2"}),
                "speaker_2_voice": (VOICES, {"default": "Puck"}),
                "project_id": ("STRING", {"forceInput": True, "tooltip": "Overrides the credentials file."}),
                "service_account": ("STRING", {"forceInput": True, "tooltip": "Overrides the credentials file."}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate_speech"
    CATEGORY = "audio/generation"

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

        return Client(
            vertexai=True,
            project=project_id.strip(),
            location=location.strip(),
            credentials=credentials,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(attempts=10, jitter=10)
            )
        )

    async def generate_speech(self, text, credentials, location, model, voice_id,
                        temperature, seed, audio_profile="", style="None", pace="None", accent="None", scene="",
                        two_speakers=False, speaker_1_name="Speaker1", speaker_2_name="Speaker2", speaker_2_voice="Puck",
                        project_id=None, service_account=None):

        if not text.strip():
            raise ValueError("Text input cannot be empty.")
        if model in GLOBAL_ONLY_MODELS and location.strip() != "global":
            raise ValueError(f"{model} is only served from global. Set location to global.")

        project_id, service_account = resolve_credentials(credentials, project_id, service_account)
        client = self.setup_client(service_account, project_id, location)

        director_parts = []
        if style not in ("", "None"): director_parts.append(f"Style: {style}")
        if pace not in ("", "None"): director_parts.append(f"Pace: {pace}")
        if accent not in ("", "None"): director_parts.append(f"Accent: {accent}")

        has_director = any([audio_profile.strip(), director_parts, scene.strip()])

        if has_director:
            sections = ["Read the following transcript based on the audio profile and director's note."]
            if audio_profile.strip():
                sections.append(f"# Audio Profile\n{audio_profile.strip()}")
            if director_parts:
                sections.append(f"# Director's note\n{'. '.join(director_parts)}.")
            if scene.strip():
                sections.append(f"## Scene:\n{scene.strip()}")
            sections.append(f"## Transcript:\n{text.strip()}")
            prompt = "\n\n".join(sections)
        else:
            prompt = f"## Transcript:\n{text.strip()}"

        def voice(name):
            return types.VoiceConfig(prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=name))

        if two_speakers:
            names = [speaker_1_name.strip(), speaker_2_name.strip()]
            if not all(names) or names[0] == names[1]:
                raise ValueError("two_speakers needs two different, non-empty speaker names.")
            for name in names:
                if not re.search(rf"^\s*{re.escape(name)}\s*:", text, re.M):
                    raise ValueError(f"two_speakers is on but no line of the text starts with '{name}:'.")
            speech_config = types.SpeechConfig(multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    types.SpeakerVoiceConfig(speaker=names[0], voice_config=voice(voice_id)),
                    types.SpeakerVoiceConfig(speaker=names[1], voice_config=voice(speaker_2_voice)),
                ]))
        else:
            speech_config = types.SpeechConfig(voice_config=voice(voice_id))

        config = types.GenerateContentConfig(
            temperature=temperature,
            seed=seed,
            response_modalities=["AUDIO"],
            speech_config=speech_config,
        )

        try:
            response = await run_blocking(
                client.models.generate_content,
                model=model,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API Error: {str(e)}")

        try:
            inline = response.candidates[0].content.parts[0].inline_data
            audio_bytes, mime_type = inline.data, inline.mime_type
        except (AttributeError, IndexError, TypeError):
            raise ValueError("API returned a response, but it contained no audio data.")

        samples, sample_rate = decode_audio(audio_bytes, mime_type)
        waveform = samples.to(torch.float32) / 32768.0
        waveform = waveform.unsqueeze(0).unsqueeze(0)

        return ({"waveform": waveform, "sample_rate": sample_rate},)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return f"{kwargs.get('text', '')}-{kwargs.get('voice_id', '')}-{kwargs.get('temperature', 1.0)}-{kwargs.get('model', '')}-{kwargs.get('seed', 69)}-{kwargs.get('audio_profile', '')}-{kwargs.get('style', '')}-{kwargs.get('pace', '')}-{kwargs.get('accent', '')}-{kwargs.get('scene', '')}-{kwargs.get('two_speakers', False)}-{kwargs.get('speaker_1_name', '')}-{kwargs.get('speaker_2_name', '')}-{kwargs.get('speaker_2_voice', '')}-{kwargs.get('location', '')}"

NODE_CLASS_MAPPINGS = {"GeminiTTSVertexNode": GeminiTTSVertexNode}
NODE_DISPLAY_NAME_MAPPINGS = {"GeminiTTSVertexNode": "Gemini TTS (Vertex AI)"}