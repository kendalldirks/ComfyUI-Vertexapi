# ComfyUI-VertexAPI

A collection of powerful custom nodes for ComfyUI that connect your local workflows to Google's Vertex AI models. Use Google's Gemini and Veo models directly within ComfyUI using your Google Cloud Vertex AI credentials.

## Key Features

All nodes run their API calls off the ComfyUI event loop, so a generation in flight doesn't stall the rest of your graph or the queue.

*   **Gemini Chat (Vertex AI):** Google's powerful multimodal AI via Vertex AI. Ask questions about an image, generate detailed descriptions, or create prompts for other models. Supports thinking budget controls.
*   **Nano Banana (Vertex AI):** Image generation and editing with the Gemini image models. The node adapts to the selected model: dropdowns list only what it supports, unused settings are greyed out, and image inputs grow as you connect them.
*   **Veo Text-to-Video (Vertex AI):** Generate high-quality video clips from text prompts using Google's Veo model via Vertex AI.
*   **Gemini TTS (Vertex AI):** Create speech from text using Google's Gemini models via Vertex AI.
*   **Gemini Diarisation (Vertex AI):** Perform speaker diarization on audio files using Gemini on Vertex AI.

---

## 🚀 Installation

1.  Navigate to your ComfyUI installation directory.
2.  Go into the `custom_nodes` folder:
    ```bash
    cd ComfyUI/custom_nodes/
    ```
3.  Clone this repository:
    ```bash
    git clone https://github.com/Aryan185/ComfyUI-VertexAPI.git
    ```
4.  Install the required Python packages:
    ```bash
    cd ComfyUI-VertexAPI
    pip install -r requirements.txt
    ```
5.  **Restart ComfyUI.**

---

## 🔑 Prerequisites: Vertex AI Setup

Most nodes in this collection require Google Cloud Vertex AI credentials to function.

You will need:
1.  **Google Cloud Project ID:** The ID of your Google Cloud project.
2.  **Location:** The region where your resources are located (e.g., `us-central1`).
3.  **Service Account JSON:** A path to your Google Cloud service account JSON key file. Ensure the service account has the necessary permissions (e.g., Vertex AI User).

---

## 🔐 Credentials

Every node has a `credentials` dropdown listing the service-account key files (`.json`) in:

```
ComfyUI/user/__vertex_credentials/
```

Pick a key and the node reads the project ID from it, with no loader node needed. New nodes start on `(select a key)` and will not run until a key is picked, so no account is billed by accident. Add a key by dropping its file in that folder and refreshing the browser.

The folder sits outside this node pack, so the pack's repo can never pick up a key. It is inside ComfyUI's git-ignored `user/` folder, and ComfyUI's userdata web API refuses `__` folders. Workflows store only the file name.

Connected `project_id` / `service_account` inputs (for example from the Vertex Credentials Loader) override the dropdown.

## 📚 Node Guide

### Gemini Chat (Vertex AI)

A versatile node for text generation and image analysis using Vertex AI.

*   **Category:** `text/generation`
*   **Inputs:**
    *   `prompt`: The text prompt.
    *   `credentials`: Key file (see Credentials). Optional `project_id` / `service_account` sockets override it.
    *   `location`: Google Cloud location (default: `global`).
    *   `model`: The Gemini model to use (default `gemini-3.8-flash`). `gemini-4-argon`, Google's top reasoning model, is also listed: it runs from `global` only, takes thinking levels low/medium/high, and ignores `temperature` and `top_p`. The Gemini 2.5 models (retiring 2026-10-20) and the `gemini-flash-latest` aliases (not served on Vertex) are no longer listed. `gemini-3.6-flash` retires 2026-11-19 and `gemini-3.7-flash` 2027-01-28.
    *   `image` (Optional): Input image for analysis.
    *   `thinking`: Enable the thinking process. Gemini 3 models cannot switch thinking off entirely, so this drops them to their lowest supported level.
    *   `thinking_level` (Optional): `auto` / `minimal` / `low` / `medium` / `high`, used by Gemini 3 and later. `auto` leaves the model on its own default. A level the chosen model doesn't support falls back to the nearest one it does.
    *   `thinking_budget` (Optional): Token budget for Gemini 2.5 and earlier. Every listed model is Gemini 3+, so it currently has no effect.
*   An empty reply (blocked prompt, safety stop, token budget spent on thinking) raises an error naming the reason instead of returning nothing.
*   **Output:**
    *   `response`: The generated text.


### Nano Banana (Vertex AI)

Image generation and editing with the Gemini image models on Vertex AI. Built on ComfyUI's V3 node schema with a small frontend script (`web/js/nano_banana_v2.js`) that makes the node follow the selected model. The per-model rules live in `vertex_model_caps.py` (`IMAGE_MODEL_CAPS`), which drives both the UI and the run-time checks.

The previous node is kept as `nano_banana_vertex.py` but is not loaded. To go back to it, import it instead of `nano_banana_vertex_v2` in `__init__.py`. Workflows built with the old node (`NanoBananaVertexNode`) need the v2 node swapped in.

*   **Category:** `image/generation`
*   **Inputs:**
    *   `credentials`: Key file (see Credentials). Optional `project_id` / `service_account` sockets override it.
    *   `model`: Nano Banana 2.1 (default, `gemini-nano-banana-2.1`), Nano Banana Pro (`gemini-3-pro-image`), Nano Banana 2 (`gemini-3.1-flash-image`), Nano Banana 2 Lite (`gemini-3.1-flash-lite-image`), Nano Banana (Gemini 2.5) (`gemini-2.5-flash-image`, retires 2027-03-15).
    *   `location`: Only `global` is offered for Nano Banana 2.1.
    *   `prompt`: Text prompt.
    *   `images`: Reference images. A new `image_N` slot appears each time one is connected, up to 14 (3 on Gemini 2.5). Every frame of a connected batch counts as an image.
    *   `aspect_ratio`: `auto` (follow the input image, or the model's default) plus the ratios the model supports. Lite has no `9:21`; Gemini 2.5 has the ten common ones only.
    *   `resolution`: The sizes the model supports. `512` is Nano Banana 2 only, Lite is `1K` only, greyed out on Gemini 2.5. Google currently ignores `image_size` on Nano Banana 2 and returns ~1K regardless.
    *   `thinking_level`: `auto` plus the model's levels: 2.1 minimal/medium/high, Nano Banana 2 minimal/high. Greyed out on Pro, Lite and Gemini 2.5.
    *   `google_search`: `off` / `web`, plus `web + image` (Google Image Search) on 2.1 and Nano Banana 2. Greyed out on Lite and Gemini 2.5.
    *   `batch_size`: 1 to 4 generations, run at the same time and returned as one image batch. Each gets its own seed and its own empty-response retries. If some fail, the rest still return, and a warning toast in the browser (plus the console log) says which failed and why. If all fail, the node errors with every reason. An image that comes back a different size from the rest is reported the same way rather than resized.
    *   `seed`: Greyed out on Nano Banana 2.1, which rejects it and is random every run. It still changes after each queue, so the node re-runs.
    *   `retry_on_empty`: On by default. When the API returns 200 with no image (a safety trip, a recitation stop), re-sends up to 3 attempts. Skipped for stops that would repeat identically (blocked prompt, prohibited content, MAX_TOKENS).
    *   Advanced: `system_instruction`, `temperature`, `top_p` (the last two greyed out on Nano Banana 2.1).
*   **Output:**
    *   `image`: The generated images, as one batch.

### Veo Text-to-Video (Vertex AI)

Generate video clips using Veo on Vertex AI.

*   **Category:** `video/generation`
*   **Inputs:**
    *   `prompt`: Video description.
    *   `credentials`, `location`: Key file (see Credentials) and Vertex location. Optional `project_id` / `service_account` sockets override the key file.
    *   `location`: `us-central1`, the only location Veo is served from.
    *   `model`: `veo-3.1-generate-001` (default), `veo-3.1-fast-generate-001`, `veo-3.1-lite-generate-001` (preview). The `-preview` IDs no longer exist on Vertex.
    *   `resolution`: `720p` / `1080p` / `4k` (`4k` on `veo-3.1-generate-001` only).
    *   `duration_seconds`: 4, 6 or 8.
    *   `number_of_videos` (Optional): 1 to 4 per request.
    *   `first_frame`, `last_frame` (Optional): Image-to-video and first/last-frame.
    *   `reference_1` to `reference_3` (Optional): Asset references (a person, character or product) to keep consistent. Not on Lite; `veo-3.1-generate-001` needs 8 seconds with references.
*   **Output:**
    *   `frames`, `audio`: Lists with one item per generated video.

### Gemini TTS (Vertex AI)

Generate speech from text using Gemini on Vertex AI.

*   **Category:** `audio/generation`
*   **Inputs:**
    *   `text`: Text to convert to speech.
    *   `credentials`, `location`: Key file (see Credentials) and Vertex location. Optional `project_id` / `service_account` sockets override the key file.
    *   `location`: Defaults to `global`. The 3.8 TTS models are served from `global` only.
    *   `model`: `gemini-3.8-flash-tts` (default), `gemini-3.8-flash-lite-tts`, `gemini-3.1-flash-tts-preview`, `gemini-2.5-pro-tts`, `gemini-2.5-flash-tts`.
    *   `voice_id`: Voice selection (speaker 1's voice in two-speaker mode).
    *   `two_speakers`, `speaker_1_name`, `speaker_2_name`, `speaker_2_voice` (Optional): Dialogue between exactly two voices. Write the text as `Name: line`, one line per turn.
    *   Inline tags such as `<short pause>`, `<sigh>` and `<cough>` can go in the text.
    *   The 3.8 models return WAV and older ones raw PCM; both are decoded, at the sample rate the API reports.
*   **Output:**
    *   `audio`: Generated audio.

### Gemini Diarisation (Vertex AI)

Speaker diarization using Gemini on Vertex AI.

*   **Category:** `audio`
*   **Inputs:**
    *   `audio`: Input audio.
    *   `num_speakers`: Number of speakers to detect.
    *   `credentials`, `location`: Key file (see Credentials) and Vertex location. Optional `project_id` / `service_account` sockets override the key file.
    *   `model`: Gemini model (same list as Gemini Chat).
*   **Output:**
    *   `speaker_1` to `speaker_4`: Audio tracks for separated speakers.

##  Acknowledgements

*   The [ComfyUI](https://github.com/comfyanonymous/ComfyUI) team.
*   Google for Vertex AI and the incredible Gemini and Veo models.
