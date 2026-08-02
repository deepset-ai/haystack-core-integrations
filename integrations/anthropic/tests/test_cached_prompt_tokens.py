"""Cached prompt tokens are billed and must survive the OpenAI-compatible conversion.

OpenAI's `prompt_tokens` INCLUDES tokens served from the prompt cache. Anthropic's
`input_tokens` is net of it, with the cached portion in `cache_read_input_tokens` and
`cache_creation_input_tokens`. `_get_openai_compatible_usage` renamed the first to the
second, which claimed OpenAI semantics for a number that did not have them.

All three are billed, so anything reading `prompt_tokens` as an OpenAI-compatible total
under-counted by the whole cached portion. On a warm cache that is most of the prompt:
`input_tokens` is by definition only the part that was NOT cached.
"""

from haystack_integrations.components.generators.anthropic.chat.utils import (
    _get_openai_compatible_usage,
)


class TestOpenAICompatibleUsage:
    def test_cache_read_tokens_are_included_in_prompt_tokens(self):
        """A warm cache: 3 uncached tokens, 20000 read from cache. All billed."""
        usage = _get_openai_compatible_usage(
            {
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 120,
                    "cache_read_input_tokens": 20000,
                }
            }
        )
        assert usage["prompt_tokens"] == 20003
        assert usage["completion_tokens"] == 120

    def test_cache_creation_tokens_are_included(self):
        """Writing the cache is billed too, at a premium."""
        usage = _get_openai_compatible_usage(
            {
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "cache_creation_input_tokens": 1500,
                }
            }
        )
        assert usage["prompt_tokens"] == 1510

    def test_both_cache_fields_are_included(self):
        usage = _get_openai_compatible_usage(
            {
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 120,
                    "cache_read_input_tokens": 20000,
                    "cache_creation_input_tokens": 1500,
                }
            }
        )
        assert usage["prompt_tokens"] == 21503

    def test_anthropic_native_keys_are_preserved(self):
        """example/prompt_caching.py reads these directly, so they must survive."""
        usage = _get_openai_compatible_usage(
            {
                "usage": {
                    "input_tokens": 3,
                    "output_tokens": 120,
                    "cache_read_input_tokens": 20000,
                    "cache_creation_input_tokens": 1500,
                }
            }
        )
        assert usage["cache_read_input_tokens"] == 20000
        assert usage["cache_creation_input_tokens"] == 1500

    def test_null_cache_fields_are_treated_as_zero(self):
        """Anthropic sends these as null rather than omitting them when unused.

        The existing fixtures in test_chat_generator.py already use
        cache_creation_input_tokens=None, so this is the common path, not an edge case.
        """
        usage = _get_openai_compatible_usage(
            {
                "usage": {
                    "input_tokens": 57,
                    "output_tokens": 40,
                    "cache_read_input_tokens": None,
                    "cache_creation_input_tokens": None,
                }
            }
        )
        assert usage["prompt_tokens"] == 57
        assert usage["completion_tokens"] == 40

    def test_zero_cache_fields_are_unchanged(self):
        """The other existing fixture shape: explicit zeros."""
        usage = _get_openai_compatible_usage(
            {
                "usage": {
                    "input_tokens": 57,
                    "output_tokens": 40,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                }
            }
        )
        assert usage["prompt_tokens"] == 57

    def test_uncached_call_is_unchanged(self):
        """With no cache in play the previous arithmetic was already right."""
        usage = _get_openai_compatible_usage({"usage": {"input_tokens": 100, "output_tokens": 50}})
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50

    def test_empty_usage_is_unchanged(self):
        assert _get_openai_compatible_usage({}) == {}
        assert _get_openai_compatible_usage({"usage": {}}) == {}
