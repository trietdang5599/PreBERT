"""Optional Stanford CoreNLP dependency-parser initialization.

Feature extraction can run without Java. In that case it still scores topic
words, but does not add their dependency-neighbour words.
"""

from __future__ import annotations

import os
import subprocess
import warnings
from pathlib import Path

from nltk.parse.stanford import StanfordDependencyParser


class DependencyParser:
    def __init__(self, model_path: str, parser_path: str) -> None:
        self.model = None
        self.disabled_reason: str | None = None

        missing = [path for path in (model_path, parser_path) if not Path(path).is_file()]
        if missing:
            self.disabled_reason = f"missing CoreNLP file(s): {', '.join(missing)}"
        else:
            try:
                subprocess.run(
                    ["java", "-version"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                    timeout=10,
                )
            except (FileNotFoundError, subprocess.SubprocessError) as exc:
                self.disabled_reason = f"Java Runtime is unavailable ({exc})"

        if self.disabled_reason is None:
            self.model = StanfordDependencyParser(
                path_to_jar=parser_path,
                path_to_models_jar=model_path,
            )
        elif os.getenv("PREBERT_REQUIRE_JAVA") == "1":
            raise RuntimeError(self.disabled_reason)
        else:
            warnings.warn(
                "Stanford dependency parsing is disabled: "
                f"{self.disabled_reason}. Continuing with topic-word sentiment "
                "only. Install a JDK to enable dependency neighbours.",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def available(self) -> bool:
        return self.model is not None

    def raw_parse(self, text: str) -> list:
        if self.model is None or not text.strip():
            return []
        try:
            parse_result = self.model.raw_parse(text)
            parsed_sentences = [list(parse.triples()) for parse in parse_result]
            return parsed_sentences[0] if parsed_sentences else []
        except Exception as exc:
            # Do not invoke a broken Java subprocess for every following review.
            self.model = None
            self.disabled_reason = f"CoreNLP failed at runtime ({exc})"
            warnings.warn(
                "Stanford dependency parsing failed and has been disabled for "
                "the rest of this run. Continuing without dependency neighbours.",
                RuntimeWarning,
                stacklevel=2,
            )
            return []


MODEL_PATH = "config/stanford-corenlp-4.5.7/stanford-corenlp-4.5.7-models.jar"
PARSER_PATH = "config/stanford-corenlp-4.5.7/stanford-corenlp-4.5.7.jar"
dep_parser = DependencyParser(MODEL_PATH, PARSER_PATH)
