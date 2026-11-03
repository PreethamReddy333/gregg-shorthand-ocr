"""A local web page: type English, see it written in Gregg shorthand as you type.

    .venv/bin/python scripts/gregg_web.py [--port 8765]

Uses the same pipeline as write_gregg.py (dictionary and phrase outlines, then render.py's
strokes), drawn as SVG in the browser. Runs only on this machine; nothing is sent anywhere.
"""

import argparse
import json
import random
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "models" / "topaz-61m"), str(ROOT / "scripts")]
from gregg_ocr.lexicon import Lexicon  # noqa: E402
from gregg_ocr.render import Style, english_items, layout  # noqa: E402
from gregg_ocr.synth import PhoneticSpeller, Renderer  # noqa: E402
from write_gregg import capitalize, detok  # noqa: E402

PAGE = Path(__file__).parent / "gregg_web.html"


def write(text: str, writer: int | None, width: float) -> dict:
    style = Style.writer(random.Random(writer)) if writer is not None else Style()
    renderer = Renderer(WORDS, phrase_rate=1.0, speller=SPELLER)
    out, longhand = [], []
    for para in (p for p in text.splitlines() if p.strip()):
        original = re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[.,?;:!]", para.replace("’", "'"), re.I)
        items = capitalize(english_items(para, renderer), original)
        longhand += [ws[0] for o, ws in items if o.lower() == ws[0].lower() and o.isalpha()]
        k = 0
        for ln in layout(" ".join(o for o, _ in items), style, width):
            outlines = []
            for d, item in zip(ln.drawings, ln.items):
                outlines.append({
                    "grascii": item,
                    "english": detok(items[k][1]),
                    "paths": [[[round(x, 3), round(y, 3)] for x, y in p] for p in d.paths],
                    "dots": [[round(x, 3), round(y, 3)] for x, y in d.dots],
                    "text": getattr(d, "text", None),
                    "pos": getattr(d, "pos", None),
                })
                k += 1
            out.append({"outlines": outlines, "english": detok([w for o in outlines for w in [o["english"]]])})
    return {"lines": out, "longhand": longhand, "pen": style.width / style.unit}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            body, kind = PAGE.read_bytes(), "text/html; charset=utf-8"
        elif url.path == "/api/write":
            q = parse_qs(url.query)
            writer = q.get("writer", [""])[0]
            try:
                result = write(q.get("text", [""])[0][:5000], int(writer) if writer else None,
                               float(q.get("width", ["26"])[0]))
            except Exception as e:  # show the problem in the page rather than a dead request
                result = {"error": f"{type(e).__name__}: {e}"}
            body, kind = json.dumps(result).encode(), "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    global WORDS, SPELLER
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    print("loading the dictionary...", flush=True)
    WORDS, SPELLER = Lexicon.load(ROOT).word_to_outline(), PhoneticSpeller()
    print(f"http://localhost:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
