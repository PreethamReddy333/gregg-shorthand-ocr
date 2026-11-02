"""Grascii + dictionary candidates -> English, with Claude reading the whole passage for context.

Topaz gives, per outline, its strokes and the dictionary's candidate words
(``I-L (while/I will/...) G (good/go)``). The ambiguity that is left (to/two/too, a misread
vowel, a name the dictionary lacks) is resolved from context, which is what the LLM is for.
The prompt is Topaz's own (models/topaz-61m/prompt.txt).

Answers are cached on disk by (model, effort, prompt, transcript), so re-running an
evaluation only pays for transcripts it hasn't seen.
"""

import hashlib
import json
import threading
from pathlib import Path

import anthropic
from pydantic import BaseModel

MODEL = "claude-opus-5-5"
# Bump when the request shape changes, so cached answers from the old shape aren't reused.
REQUEST_VERSION = 2


class Transcription(BaseModel):
    """Structured output keeps the answer separate from any working-out on hard lines."""

    english: str


class EnglishDecoder:
    def __init__(self, prompt: str, cache_path: Path, model: str = MODEL, effort: str = "medium"):
        self.client = anthropic.Anthropic()
        self.prompt = prompt
        self.model = model
        self.effort = effort
        self.cache_path = cache_path
        self.cache = {}
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                row = json.loads(line)
                self.cache[row["key"]] = row["english"]
        self._lock = threading.Lock()
        self.refusals = 0

    def _key(self, transcript: str) -> str:
        blob = json.dumps([REQUEST_VERSION, self.model, self.effort, self.prompt, transcript])
        return hashlib.sha256(blob.encode()).hexdigest()

    def decode(self, transcript: str) -> str:
        """Topaz intermediate text (one passage, lines separated by newlines) -> English."""
        key = self._key(transcript)
        if key in self.cache:
            return self.cache[key]

        if not transcript.strip():
            return ""
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=self.prompt,
            messages=[{"role": "user", "content": f"Transcript:\n{transcript}"}],
            output_config={"effort": self.effort},
            output_format=Transcription,
            # On a safety decline, the API re-runs the request on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal" or response.parsed_output is None:
            with self._lock:
                self.refusals += 1
            return ""  # not cached, so a later run retries it
        english = response.parsed_output.english.strip()

        with self._lock:
            self.cache[key] = english
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a") as f:
                f.write(json.dumps({"key": key, "transcript": transcript, "english": english}) + "\n")
        return english
