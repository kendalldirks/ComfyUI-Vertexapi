"""Shared thread pool for the blocking google-genai calls.

ComfyUI runs node functions on its asyncio event loop. A synchronous HTTP call
to Vertex blocks that loop for the whole round trip, which stalls every other
node in the queue. Declaring the node's FUNCTION as `async def` and awaiting
`run_blocking(...)` hands the call to a worker thread instead, so independent
branches of a graph — and other queued prompts — keep running while a
generation is in flight.

Its own module so it cannot conflict with upstream changes.
"""

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor

# Shared across every node in the pack, so a graph with several Vertex nodes
# does not spawn a pool each.
_EXECUTOR = None
MAX_WORKERS = 10


def get_executor():
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(
            max_workers=MAX_WORKERS, thread_name_prefix="vertexapi"
        )
    return _EXECUTOR


async def run_blocking(fn, *args, **kwargs):
    """Await `fn(*args, **kwargs)` on the shared pool instead of the event loop."""
    if kwargs:
        fn = functools.partial(fn, **kwargs)
    return await asyncio.get_event_loop().run_in_executor(get_executor(), fn, *args)
