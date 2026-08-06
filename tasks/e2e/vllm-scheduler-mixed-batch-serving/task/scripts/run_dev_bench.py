#!/usr/bin/env python3
"""Public dev benchmark — launches /app/submission/launch_server.sh, sends a few PUBLIC
requests, reports rough latency. The hidden verifier uses DIFFERENT workloads and more
iterations, so a good dev number here does not guarantee the score. Use it only to check
your server starts and to get a coarse latency signal while iterating.

    python3 /app/run_dev_bench.py                 # launch + bench + shutdown
    python3 /app/run_dev_bench.py --no-server --port 30001   # bench an already-running server
"""
import argparse, json, os, signal, subprocess, time, statistics, urllib.request

PUBLIC_PROMPTS = [
    ("Write a short paragraph about the ocean.", 128),
    ("List the numbers from 1 to 20, then list them again in reverse.", 128),
    ("Explain what a hash table is in two sentences.", 96),
]


def send(port, content, max_tokens):
    body = json.dumps({"model": "default", "messages": [{"role": "user", "content": content}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=300)
    dt = (time.perf_counter() - t) * 1000.0
    json.loads(resp.read())
    return dt


def wait_health(port, proc, timeout=1800):
    dl = time.time() + timeout
    while time.time() < dl:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError("server exited early")
        try:
            r = subprocess.run(["curl", "-sS", "-o", "/dev/null", "-m", "4", "-w", "%{http_code}",
                                f"http://localhost:{port}/health"], capture_output=True, text=True, timeout=6)
            if r.stdout.strip() == "200":
                return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError("server never became healthy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30001)
    ap.add_argument("--no-server", action="store_true")
    a = ap.parse_args()
    proc = None
    try:
        if not a.no_server:
            env = {**os.environ, "PORT": str(a.port), "MODEL_PATH": "/app/model"}
            proc = subprocess.Popen(["bash", "/app/submission/launch_server.sh"], env=env, preexec_fn=os.setsid)
            wait_health(a.port, proc)
        for content, mt in PUBLIC_PROMPTS:
            lat = [send(a.port, content, mt) for _ in range(5)]
            print(f"  {content[:40]!r}: median {statistics.median(lat):.1f} ms")
        print("dev bench complete")
    finally:
        if proc is not None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass


if __name__ == "__main__":
    main()
