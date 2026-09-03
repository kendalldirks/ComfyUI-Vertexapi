"""Pull the generated image out of a Gemini response, and explain it when there
isn't one.

A 200 OK with no image is normal and has several distinct causes — a blocked
prompt, a safety stop, the output budget spent on thinking, or the model
answering in text instead. The SDK expresses all of them as
`candidates[0].content.parts` being None, so indexing straight into it turns
every one of them into the same opaque TypeError. This reads whatever the
response actually carries and reports the reason.

Its own module so it cannot conflict with upstream changes.
"""

# finish_reason values that mean "a filter stopped this", with what to do.
_FILTER_ADVICE = {
    "SAFETY": "a safety filter stopped the generation",
    "IMAGE_SAFETY": "the generated image was rejected by the image safety filter",
    "PROHIBITED_CONTENT": "the request was refused as prohibited content",
    "RECITATION": "generation stopped because the output was reciting training data",
    "BLOCKLIST": "the request hit a term blocklist",
    "SPII": "the request was refused for containing sensitive personal information",
}


def _name(value):
    """finish_reason / block_reason may be an enum or a plain string."""
    if value is None:
        return None
    return getattr(value, "name", None) or str(value)


def extract_image_bytes(response, model=""):
    """Return the first inline image's bytes, or raise ValueError explaining why
    there is none."""
    candidates = getattr(response, "candidates", None) or []

    texts = []
    for cand in candidates:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None) if inline is not None else None
            if data:
                return data
            text = getattr(part, "text", None)
            if text and text.strip():
                texts.append(text.strip())

    raise ValueError(_explain(response, model, candidates, texts))


def _explain(response, model, candidates, texts):
    where = f"{model} returned a response with no image" if model else "The response contained no image"
    bits = []

    # 1. Prompt rejected before anything was generated: candidates is empty.
    feedback = getattr(response, "prompt_feedback", None)
    block = _name(getattr(feedback, "block_reason", None)) if feedback is not None else None
    if block:
        bits.append(f"the prompt was blocked before generation (block_reason={block})")
        msg = getattr(feedback, "block_reason_message", None)
        if msg:
            bits.append(f'the API said: "{msg}"')
    elif not candidates:
        bits.append("the API returned no candidates at all")

    # 2. Otherwise the candidate itself says why it stopped.
    for cand in candidates[:1]:
        reason = _name(getattr(cand, "finish_reason", None))
        if reason and reason not in ("STOP", "FINISH_REASON_UNSPECIFIED"):
            bits.append(f"{_FILTER_ADVICE.get(reason, 'generation stopped early')} (finish_reason={reason})")
            if reason == "MAX_TOKENS":
                bits.append(
                    "the output token budget ran out before an image was produced — "
                    "thinking tokens count against it, so try a lower thinking level, "
                    "a shorter prompt, or a model with a larger output budget "
                    "(gemini-3.1-flash-lite-image allows only 4,096)"
                )
        elif reason == "STOP" and texts:
            bits.append("the model replied with text instead of an image")

        detail = getattr(cand, "finish_message", None)
        if detail:
            bits.append(f'the API said: "{detail}"')

        flagged = [
            f"{_name(getattr(r, 'category', None))}={_name(getattr(r, 'probability', None))}"
            for r in getattr(cand, "safety_ratings", None) or []
            if _name(getattr(r, "probability", None)) not in (None, "NEGLIGIBLE", "HARM_PROBABILITY_UNSPECIFIED")
        ]
        if flagged:
            bits.append("safety ratings above negligible: " + ", ".join(flagged))

    if texts:
        joined = " ".join(texts)
        bits.append(f'text returned: "{joined[:400]}{"..." if len(joined) > 400 else ""}"')

    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        counts = {
            k: getattr(usage, k, None)
            for k in ("prompt_token_count", "candidates_token_count", "thoughts_token_count", "total_token_count")
        }
        counts = {k: v for k, v in counts.items() if v}
        if counts:
            bits.append("tokens: " + ", ".join(f"{k.replace('_token_count', '')}={v}" for k, v in counts.items()))

    if not bits:
        bits.append("no reason was given — this is usually a transient filter hit, so retry with a different seed")

    return where + ": " + "; ".join(bits) + "."
