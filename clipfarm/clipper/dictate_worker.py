"""The live-dictation worker: a long-running process in Parakeet's own environment (parakeet_mlx is not installed anywhere else).

It loads the model once and then speaks one JSON object per line on stdin/stdout, so the engine can feed it the new audio every second
or so and get back the words so far:

    {"op": "start"}                      a new dictation          -> {"ok": true}
    {"op": "feed", "pcm": "<base64>"}    16 kHz mono int16 audio  -> {"stable": "words that are final", "tail": "words still changing"}
    {"op": "stop"}                       end of that dictation    -> {"ok": true}
    {"op": "quit"}                       exit

Stdlib plus parakeet_mlx / mlx only (no clipfarm imports: it runs under another interpreter).
"""

import base64
import json
import sys

MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
CONTEXT = (256, 48)                 # (left, right) attention frames of 80 ms: a short look-ahead keeps finished words close to what was just said


def say(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def text_of(tokens) -> str:
    return "".join(str(getattr(t, "text", "")) for t in tokens)


def main() -> None:
    import mlx.core as mx
    import numpy as np
    from parakeet_mlx import from_pretrained

    model = from_pretrained(MODEL)
    say({"ready": True})
    ctx = stream = None
    for line in sys.stdin:
        try:
            msg = json.loads(line)
            op = msg.get("op")
            if op == "quit":
                break
            if op in ("start", "stop"):
                if ctx is not None:
                    ctx.__exit__(None, None, None)
                    ctx = stream = None
                if op == "start":
                    ctx = model.transcribe_stream(context_size=CONTEXT)
                    stream = ctx.__enter__()
                say({"ok": True})
            elif op == "feed":
                if stream is None:
                    say({"error": "no dictation started"})
                    continue
                pcm = np.frombuffer(base64.b64decode(msg.get("pcm") or ""), dtype=np.int16).astype(np.float32) / 32768.0
                if len(pcm):
                    stream.add_audio(mx.array(pcm))
                say({"stable": text_of(stream.finalized_tokens), "tail": text_of(stream.draft_tokens)})
            else:
                say({"error": f"unknown op {op}"})
        except Exception as e:  # noqa: BLE001 - the engine falls back to the ordinary dictation
            say({"error": f"{type(e).__name__}: {e}"[:300]})


if __name__ == "__main__":
    main()
