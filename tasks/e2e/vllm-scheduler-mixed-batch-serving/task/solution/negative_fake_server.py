#!/usr/bin/env python3
"""REVIEWER-ONLY negative: a FAST but WRONG "server".

Never baked, never part of tests/, never run at scoring. It exists so validation can prove
the perf-class pre-gates BITE: a submission that is much faster than the baseline but does
not actually run the model must score 0 with a NAMED reason, not a high reward.

It speaks just enough of the OpenAI chat-completions API to satisfy a naive readiness probe
and answers instantly with canned text, ignoring the requested token budget. Expected
verdict: correctness_failed (token parity) — and, if a candidate ever managed to match the
reference text without paying for it, cheating_detected via the decode-work probe.
"""
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CANNED = ("The answer depends on several factors, and a careful analysis shows that the "
          "result follows from the premises stated above.")


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._json({"object": "list", "data": [{"id": "default", "object": "model"}]})
        elif self.path.startswith("/health"):
            self._json({"status": "ok"})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            pass
        self._json({
            "id": "fake", "object": "chat.completion", "created": int(time.time()),
            "model": "default",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": CANNED}}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 24, "total_tokens": 32},
        })

    def log_message(self, *a):
        return


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "30001"))
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
