"""Turn a ComfyUI VIDEO input into inline bytes the Gemini image models accept.

Imports from comfy_api are soft: on a ComfyUI old enough to lack the native
VIDEO type the pack still loads, and the video inputs simply refuse to run.
"""

import io
import os

try:  # ComfyUI >= the native VIDEO type
    from comfy_api.util import VideoContainer
    _HAVE_COMFY_VIDEO = True
except Exception:  # pragma: no cover - depends on the host ComfyUI
    VideoContainer = None
    _HAVE_COMFY_VIDEO = False

# Container extensions Vertex accepts directly, so they can be passed through
# without a re-mux.
_EXT_MIME = {
    "mp4": "video/mp4",
    "m4v": "video/mp4",
    "mov": "video/quicktime",
    "webm": "video/webm",
    "mpeg": "video/mpeg",
    "mpg": "video/mpeg",
    "flv": "video/x-flv",
    "wmv": "video/wmv",
    "3gp": "video/3gpp",
}

# Vertex takes inline media as base64 inside the request body. Well before the
# 500 MB account limit the request itself gets unwieldy, so warn early.
INLINE_WARN_BYTES = 15 * 1024 * 1024


def video_to_bytes(video, label="video"):
    """Return (data, mime_type) for a ComfyUI VideoInput.

    Passes the file through untouched when its container is already one Vertex
    accepts; otherwise re-muxes to MP4.
    """
    source = video.get_stream_source()

    if isinstance(source, str):
        ext = os.path.splitext(source)[1].lower().lstrip(".")
        mime = _EXT_MIME.get(ext)
        if mime:
            with open(source, "rb") as fh:
                data = fh.read()
            return _checked(data, mime, label)

    elif isinstance(source, io.IOBase):
        source.seek(0)

    # Unknown or in-memory container: normalise to MP4.
    if not _HAVE_COMFY_VIDEO:
        raise RuntimeError(
            "This ComfyUI build has no comfy_api.util.VideoContainer, so a video "
            "in an unrecognised container cannot be converted. Feed the node an "
            "mp4, mov, webm, mpeg, flv, wmv or 3gp file instead."
        )
    buf = io.BytesIO()
    video.save_to(buf, format=VideoContainer.MP4)
    return _checked(buf.getvalue(), "video/mp4", label)


def _checked(data, mime, label):
    if len(data) > INLINE_WARN_BYTES:
        print(
            f"[VertexAPI] {label} is {len(data) / 1024 / 1024:.1f} MB and is sent "
            f"inline in the request; large videos may be rejected or time out. "
            f"Trim it first if the call fails."
        )
    return data, mime
