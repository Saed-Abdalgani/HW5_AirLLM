"""Unit tests for services.backends.base (InferenceBackend ABC)."""

from __future__ import annotations

import pytest

from airllm_bench.services.backends.base import InferenceBackend


class TestInferenceBackendABC:
    def test_cannot_instantiate_incomplete_subclass(self):
        """A subclass missing generate() cannot be instantiated."""

        class Incomplete(InferenceBackend):
            @property
            def name(self) -> str:
                return "incomplete"

            def load(self) -> None:
                pass

            # generate() deliberately omitted

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_minimal_concrete_subclass(self):
        """A fully implemented subclass can be instantiated and called."""

        class Concrete(InferenceBackend):
            @property
            def name(self) -> str:
                return "concrete"

            def load(self) -> None:
                pass

            def generate(self, prompt: str, max_new_tokens: int) -> str:
                return f"echo:{prompt}"

        b = Concrete()
        b.load()
        assert b.generate("hello", 8) == "echo:hello"

    def test_teardown_is_no_op_by_default(self):
        """teardown() should not raise if not overridden."""

        class Minimal(InferenceBackend):
            @property
            def name(self) -> str:
                return "minimal"

            def load(self) -> None:
                pass

            def generate(self, prompt: str, max_new_tokens: int) -> str:
                return ""

        b = Minimal()
        b.teardown()  # should not raise
