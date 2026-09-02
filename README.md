# ComfyUI-VertexAPI

A collection of powerful custom nodes for ComfyUI that connect your local workflows to Google's Vertex AI models. Use Google's Gemini and Veo models directly within ComfyUI using your Google Cloud Vertex AI credentials.

## Key Features

All nodes run their API calls off the ComfyUI event loop, so a generation in flight doesn't stall the rest of your graph or the queue.

*   **Gemini Chat (Vertex AI):** Google's powerful multimodal AI via Vertex AI. Ask questions about an image, generate detailed descriptions, or create prompts for other models. Supports thinking budget controls.
*   **Nano Banana (Vertex AI):** A creative image generation node using a specialized Gemini model on Vertex AI.
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

## 📚 Node Guide

### Gemini Chat (Vertex AI)

A versatile node for text generation and image analysis using Vertex AI.

*   **Category:** `text/generation`
*   **Inputs:**
    *   `prompt`: The text prompt.
    *   `project_id`: Your Google Cloud Project ID.
    *   `location`: Google Cloud location (default: `us-central1`).
    *   `service_account`: Path to your service account JSON file.
    *   `model`: The Gemini model to use (e.g., `gemini-3.7-flash`).
    *   `image` (Optional): Input image for analysis.
    *   `thinking`: Enable the thinking process. Gemini 3 models cannot switch thinking off entirely, so this drops them to their lowest supported level.
    *   `thinking_level` (Optional): `auto` / `minimal` / `low` / `medium` / `high`, used by Gemini 3 and later. `auto` leaves the model on its own default. A level the chosen model doesn't support falls back to the nearest one it does.
    *   `thinking_budget` (Optional): Token budget, used by Gemini 2.5 and earlier. Clamped to each model's supported range. Ignored by Gemini 3 models, which take `thinking_level` instead — sending both is an API error.
*   **Output:**
    *   `response`: The generated text.


### Nano Banana (Vertex AI)

Creative image generation using Gemini on Vertex AI.

*   **Category:** `image/generation`
*   **Inputs:**
    *   `project_id`, `location`, `service_account`: Vertex AI credentials.
    *   `model`: Gemini image model (e.g., `gemini-3-pro-image`).
    *   `aspect_ratio`: The full set of 15 ratios. Which ones are valid depends on the model — Nano Banana 2 Lite has no `9:21`, and the legacy `gemini-2.5-flash-image` supports only the ten common ones. An unsupported combination raises an error naming what the model does accept.
    *   `resolution`: `512` / `1K` / `2K` / `4K`. `512` is Nano Banana 2 only, Lite is `1K` only, and `gemini-2.5-flash-image` has no resolution control at all (the value is ignored). Note that Google currently ignores `image_size` on `gemini-3.1-flash-image` and returns ~1K regardless.
    *   `google_search`: Grounding. Supported by Nano Banana Pro and Nano Banana 2 only; ignored with a log line on the others.
    *   `prompt` (Optional): Text prompt.
    *   `image_1` to `image_14` (Optional): Source images. The Gemini 3 image models accept up to 14 per request, `gemini-2.5-flash-image` only 3.
    *   `video_1`, `video_2` (Optional): Source videos, for generating an image from video. Nano Banana 2 and Nano Banana 2 Lite only — Pro and the legacy 2.5 model take no video input. The API allows 10 per request, but they are sent inline in the request body, so two is the practical ceiling.
    *   `youtube_url` (Optional): A single YouTube URL used as video input, on the same models.
*   **Output:**
    *   `image`: Generated image.

### Veo Text-to-Video (Vertex AI)

Generate video clips using Veo on Vertex AI.

*   **Category:** `video/generation`
*   **Inputs:**
    *   `prompt`: Video description.
    *   `project_id`, `location`, `service_account`: Vertex AI credentials.
    *   `model`: Veo model (e.g., `veo-2.0-generate-001`).
*   **Output:**
    *   `frames`: Generated video frames.

### Gemini TTS (Vertex AI)

Generate speech from text using Gemini on Vertex AI.

*   **Category:** `audio/generation`
*   **Inputs:**
    *   `text`: Text to convert to speech.
    *   `project_id`, `location`, `service_account`: Vertex AI credentials.
    *   `model`: Gemini TTS model.
    *   `voice_id`: Voice selection.
*   **Output:**
    *   `audio`: Generated audio.

### Gemini Diarisation (Vertex AI)

Speaker diarization using Gemini on Vertex AI.

*   **Category:** `audio`
*   **Inputs:**
    *   `audio`: Input audio.
    *   `num_speakers`: Number of speakers to detect.
    *   `project_id`, `location`, `service_account`: Vertex AI credentials.
    *   `model`: Gemini model.
*   **Output:**
    *   `speaker_1` to `speaker_4`: Audio tracks for separated speakers.

##  Acknowledgements

*   The [ComfyUI](https://github.com/comfyanonymous/ComfyUI) team.
*   Google for Vertex AI and the incredible Gemini and Veo models.
