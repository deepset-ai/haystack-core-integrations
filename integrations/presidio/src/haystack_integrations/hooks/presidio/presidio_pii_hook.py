# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from itertools import repeat
from typing import Any, ClassVar, Literal, get_args

from haystack import logging, tracing
from haystack.components.agents.state import State
from haystack.components.agents.state.state_utils import replace_values
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole, TextContent, ToolCallResult
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine

from haystack_integrations.common.presidio.utils import SPACY_DEFAULT_MODELS as _SPACY_DEFAULT_MODELS
from haystack_integrations.common.presidio.utils import build_analyzer_engine

logger = logging.getLogger(__name__)

PIIAction = Literal["redact", "block", "stop"]
PIIErrorAction = Literal["pass", "block", "stop"]

# Marks a message the hook has already checked, so registering the hook at several points checks each message once.
_CHECKED_META_KEY = "pii_checked"
# Same key Haystack's own hooks use to mark synthetic reminders, which compaction then treats as non-task messages.
_SYSTEM_REMINDER_META_KEY = "system_reminder"
_STOP_REASON = "pii_detected"
_SEVERITY: dict[str, int] = {"none": 0, "pass": 0, "redact": 1, "block": 2, "stop": 3}
_DEFAULT_BLOCK_MESSAGE = "[This message was removed because it contained sensitive information.]"
_PII_REMINDER = (
    "A message was removed because it contained sensitive information ({entities}). "
    "Do not ask for, repeat, or guess this information. If the user provided it, tell them it cannot be accepted."
)
_ERROR_REMINDER = (
    "A message was removed because it could not be checked for sensitive information. "
    "Continue without it, and do not ask for it to be repeated."
)


def _map_texts(message: ChatMessage, fn: Callable[[str], str]) -> list[Any]:
    """
    Apply `fn` to every text the hook checks in `message` and return the rebuilt content.

    Checked texts are `TextContent` parts and tool results, both plain-string results and their `TextContent` parts.
    Tool calls, reasoning, images, and files are returned unchanged.

    :param message: The message whose content to rebuild.
    :param fn: Maps a checked text to its replacement.
    :returns: The message's content with every checked text replaced by `fn(text)`.
    """
    content: list[Any] = []
    # ChatMessage exposes its parts only grouped by type, so `_content` is the one way to rebuild them in order.
    for part in message._content:
        if isinstance(part, TextContent):
            content.append(TextContent(text=fn(part.text)))
        elif isinstance(part, ToolCallResult):
            if isinstance(part.result, str):
                new_result: Any = fn(part.result)
            else:
                new_result = [TextContent(text=fn(p.text)) if isinstance(p, TextContent) else p for p in part.result]
            content.append(replace(part, result=new_result))
        else:
            content.append(part)
    return content


def _checked_texts(message: ChatMessage) -> list[str]:
    """Return the texts `_map_texts` would rewrite in `message`, in the same order."""
    texts: list[str] = []

    def collect(text: str) -> str:
        texts.append(text)
        return text

    _map_texts(message=message, fn=collect)
    return texts


class PresidioPIIHook:
    """
    Detects personally identifiable information (PII) in an Agent's messages with Microsoft Presidio.

    Register the same instance at `before_llm`, so PII never reaches the LLM provider, and at `after_run`, so the final
    reply returned to the user is checked too. The final reply is produced by the last LLM call, so no `before_llm`
    hook runs after it:

    <!-- test-ignore -->
    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack_integrations.hooks.presidio import PresidioPIIHook

    pii_hook = PresidioPIIHook(
        entity_score_thresholds={"PERSON": 0.6},
        entity_actions={"CREDIT_CARD": "block", "IBAN_CODE": "stop"},
    )
    agent = Agent(
        chat_generator=OpenAIChatGenerator(),
        hooks={"before_llm": [pii_hook], "after_run": [pii_hook]},
    )
    result = agent.run(messages=[ChatMessage.from_user("Hi, I'm John Smith from Berlin.")])
    # The LLM receives: "Hi, I'm <PERSON> from <LOCATION>."
    ```

    Each run checks the messages it has not checked yet: user and assistant text and tool results. System messages,
    system reminders, tool-call arguments, reasoning, images, and files are not checked. Every detected entity maps to
    an action, and a message takes the strictest action among its entities:

    - `"redact"` (default): Presidio's anonymizer replaces each entity with a placeholder such as `<PERSON>`.
    - `"block"`: The message's text is replaced with `block_message`. Mid-run, a system reminder is added telling the
      model the message was removed; once the run is exiting, the reply is replaced without a reminder.
    - `"stop"`: The message is redacted and the run stops before the next LLM call, with `exit_reason` set to
      `"pii_detected"`.

    Limitations:

    - When the Agent or its chat generator has a `streaming_callback`, the reply streams to the user before any hook
      runs, so PII in the reply cannot be stopped in that mode.
    - Because the model only sees placeholders, the tool-call arguments it writes contain placeholders too, so a tool
      that needs the real value (for example, an email address to send to) receives `<EMAIL_ADDRESS>`.

    Each run records its decision on the hook's tracing span under `haystack.agent.hook.pii.*`: how many messages it
    checked, whether anything was found, the action taken, the count of each entity type, and whether Presidio failed.
    The checked and rewritten texts are recorded as content tags, so they are only emitted when content tracing is on.
    """

    allowed_hook_points: ClassVar[tuple[str, ...]] = ("before_run", "before_llm", "after_tool", "after_run")

    SPACY_DEFAULT_MODELS: ClassVar[dict[str, str]] = _SPACY_DEFAULT_MODELS
    """Mapping from ISO 639-1 language code to the largest available spaCy model for that language.

    Used to automatically select an NLP model when `models` is not specified.
    See [spaCy documentation](https://spacy.io/models) for the full list of available spaCy models.
    """

    def __init__(
        self,
        *,
        language: str = "en",
        entities: list[str] | None = None,
        score_threshold: float = 0.35,
        entity_score_thresholds: dict[str, float] | None = None,
        entity_actions: dict[str, PIIAction] | None = None,
        on_error: PIIErrorAction = "block",
        block_message: str = _DEFAULT_BLOCK_MESSAGE,
        models: list[dict[str, str]] | None = None,
    ) -> None:
        """
        Initialize the hook.

        :param language: ISO 639-1 language code for PII detection. Defaults to `"en"`.
            For languages in `SPACY_DEFAULT_MODELS`, the spaCy model is loaded automatically at warm-up time.
            See [Presidio supported languages](https://microsoft.github.io/presidio/analyzer/languages/).
        :param entities: PII entity types to detect, for example `["PERSON", "PHONE_NUMBER"]`.
            If `None`, all entity types Presidio supports for `language` are detected.
            See [Presidio supported entities](https://microsoft.github.io/presidio/supported_entities/).
        :param score_threshold: Minimum confidence score (0-1) for a detected entity to count, for entity types not
            listed in `entity_score_thresholds`. Defaults to `0.35`.
        :param entity_score_thresholds: Per-entity-type minimum confidence scores (0-1) that override
            `score_threshold`, for example `{"PERSON": 0.6}` to ignore low-confidence name matches.
        :param entity_actions: Per-entity-type actions, one of `"redact"`, `"block"`, or `"stop"`, for example
            `{"CREDIT_CARD": "block"}`. Entity types not listed are redacted.
        :param on_error: What to do with a message when Presidio raises an error while checking it:
            `"block"` (default) replaces it with `block_message`, `"stop"` also stops the run, and `"pass"` leaves
            the message unchanged, which can let PII through.
        :param block_message: The text that replaces the text of a blocked message.
        :param models: Advanced override: spaCy model configurations, each with `"lang_code"` and `"model_name"`
            keys, for example `[{"lang_code": "fr", "model_name": "fr_core_news_md"}]`. If `None`, the model is
            selected from `SPACY_DEFAULT_MODELS` based on `language`.
        :raises ValueError: If a score threshold is outside 0-1, or an action or `on_error` value is unknown.
        """
        thresholds = {"score_threshold": score_threshold, **(entity_score_thresholds or {})}
        for name, value in thresholds.items():
            if not 0 <= value <= 1:
                msg = f"Score thresholds must be between 0 and 1, got {value} for '{name}'."
                raise ValueError(msg)
        for entity, action in (entity_actions or {}).items():
            if action not in get_args(PIIAction):
                msg = f"Unknown action '{action}' for entity '{entity}'. Valid actions are: redact, block, stop."
                raise ValueError(msg)
        if on_error not in get_args(PIIErrorAction):
            msg = f"Unknown `on_error` value '{on_error}'. Valid values are: pass, block, stop."
            raise ValueError(msg)

        self.language = language
        self.entities = entities
        self.score_threshold = score_threshold
        self.entity_score_thresholds = entity_score_thresholds or {}
        self.entity_actions = entity_actions or {}
        self.on_error = on_error
        self.block_message = block_message
        self.models = models
        self._analyzer: AnalyzerEngine | None = None
        self._anonymizer: AnonymizerEngine | None = None

    def warm_up(self) -> None:
        """
        Load the Presidio analyzer and anonymizer engines and check the configured entity types.

        The Agent calls this from its own `warm_up`. Calling it again does nothing.

        :raises ValueError: If no spaCy model is available for `language`, or an entity type in `entities`,
            `entity_score_thresholds`, or `entity_actions` is not supported for `language`, or an entity type in
            `entity_score_thresholds` or `entity_actions` is not in `entities`.
        """
        if self._analyzer is not None:
            return

        analyzer = build_analyzer_engine(
            language=self.language, models=self.models, default_models=self.SPACY_DEFAULT_MODELS
        )
        self._validate_entities(supported=set(analyzer.get_supported_entities(language=self.language)))
        self._anonymizer = AnonymizerEngine()
        self._analyzer = analyzer

    def run(self, state: State) -> None:
        """
        Check the messages not checked yet and redact, block, or stop as configured.

        :param state: The Agent's live `State`. Its `messages` are rewritten in place; `stop_run` is set when a message
            triggers `"stop"`.
        """
        if self._analyzer is None:
            self.warm_up()

        messages: list[ChatMessage] = state.data.get("messages") or []
        to_check = [
            index
            for index, message in enumerate(messages)
            if message.role != ChatRole.SYSTEM
            and _CHECKED_META_KEY not in message.meta
            and _SYSTEM_REMINDER_META_KEY not in message.meta
        ]
        if not to_check:
            return

        # Exit hooks run after the final reply; a reminder appended then would become the Agent's last message.
        mid_run = state.data.get("exit_reason") is None
        rewritten = list(messages)
        reminders: list[ChatMessage] = []
        entity_counts: Counter[str] = Counter()
        run_action = "none"
        errored = False
        inputs: list[str] = []
        outputs: list[str] = []

        for index in to_check:
            message = messages[index]
            texts = _checked_texts(message=message)
            inputs.extend(texts)

            try:
                findings = [self._find_entities(text=text) for text in texts]
            except Exception as error:
                logger.warning(
                    "Presidio failed to check a message for PII; applying on_error='{on_error}'. Error: {error}",
                    on_error=self.on_error,
                    error=error,
                )
                errored = True
                # Nothing was detected, so there is nothing to redact: a failed message is blocked or left as is.
                action: str = self.on_error
                if action != "pass":
                    rewritten[index] = self._replace_content(message=message, replacements=repeat(self.block_message))
                    if mid_run:
                        reminders.append(self._reminder(text=_ERROR_REMINDER))
            else:
                found = Counter(result.entity_type for results in findings for result in results)
                entity_counts.update(found)
                action = max(
                    (self.entity_actions.get(entity, "redact") for entity in found),
                    key=_SEVERITY.__getitem__,
                    default="none",
                )
                if action == "block":
                    rewritten[index] = self._replace_content(message=message, replacements=repeat(self.block_message))
                    if mid_run:
                        entity_list = ", ".join(sorted(found))
                        reminders.append(self._reminder(text=_PII_REMINDER.format(entities=entity_list)))
                elif action in ("redact", "stop"):
                    anonymized = [self._anonymize(text=t, results=r) for t, r in zip(texts, findings, strict=True)]
                    rewritten[index] = self._replace_content(message=message, replacements=anonymized)

            if action == "stop":
                state.set("stop_run", _STOP_REASON)
            run_action = max(run_action, action, key=_SEVERITY.__getitem__)
            if rewritten[index] is message:
                rewritten[index] = replace(message, _meta={**message.meta, _CHECKED_META_KEY: True})
            outputs.extend(_checked_texts(message=rewritten[index]))

        state.set("messages", rewritten + reminders, handler_override=replace_values)
        self._trace(
            checked=len(to_check),
            action=run_action,
            entity_counts=dict(entity_counts),
            errored=errored,
            inputs=inputs,
            outputs=outputs,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the hook to a dictionary.

        :returns: The serialized hook.
        """
        return default_to_dict(
            self,
            language=self.language,
            entities=self.entities,
            score_threshold=self.score_threshold,
            entity_score_thresholds=self.entity_score_thresholds,
            entity_actions=self.entity_actions,
            on_error=self.on_error,
            block_message=self.block_message,
            models=self.models,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PresidioPIIHook":
        """
        Deserialize the hook from a dictionary.

        :param data: The dictionary produced by `to_dict`.
        :returns: The deserialized hook.
        """
        return default_from_dict(cls, data)

    def _validate_entities(self, supported: set[str]) -> None:
        """Raise if a configured entity type would never be detected, so a typo fails at warm-up instead of silently."""
        configured = {
            "entities": set(self.entities or []),
            "entity_score_thresholds": set(self.entity_score_thresholds),
            "entity_actions": set(self.entity_actions),
        }
        for param, names in configured.items():
            if unknown := sorted(names - supported):
                msg = (
                    f"`{param}` contains entity types not supported for language '{self.language}': "
                    f"{', '.join(unknown)}. Supported entity types: {', '.join(sorted(supported))}."
                )
                raise ValueError(msg)
        if self.entities is not None:
            for param in ("entity_score_thresholds", "entity_actions"):
                if undetected := sorted(configured[param] - configured["entities"]):
                    msg = f"`{param}` contains entity types missing from `entities`: {', '.join(undetected)}."
                    raise ValueError(msg)

    def _find_entities(self, text: str) -> list[RecognizerResult]:
        """
        Detect entities in `text` and keep those above their entity type's threshold.

        Presidio takes a single threshold, so the analyzer runs at the lowest configured one and the per-entity
        thresholds are applied afterwards.
        """
        lowest = min([self.score_threshold, *self.entity_score_thresholds.values()])
        results = self._analyzer.analyze(  # type: ignore[union-attr]
            text=text, language=self.language, entities=self.entities, score_threshold=lowest
        )
        return [r for r in results if r.score >= self.entity_score_thresholds.get(r.entity_type, self.score_threshold)]

    def _anonymize(self, text: str, results: list[RecognizerResult]) -> str:
        """Replace each detected entity in `text` with its entity type placeholder."""
        if not results:
            return text
        return self._anonymizer.anonymize(text=text, analyzer_results=results).text  # type: ignore[union-attr, arg-type]

    @staticmethod
    def _replace_content(message: ChatMessage, replacements: Iterable[str]) -> ChatMessage:
        """Return a copy of `message`, marked as checked, with its checked texts replaced in order by `replacements`."""
        replacement_iter = iter(replacements)
        content = _map_texts(message=message, fn=lambda _: next(replacement_iter))
        return replace(message, _content=content, _meta={**message.meta, _CHECKED_META_KEY: True})

    @staticmethod
    def _reminder(text: str) -> ChatMessage:
        """Build a system reminder the hook itself will skip on later runs."""
        return ChatMessage.from_user(text, meta={_SYSTEM_REMINDER_META_KEY: True, _CHECKED_META_KEY: True})

    @staticmethod
    def _trace(
        *,
        checked: int,
        action: str,
        entity_counts: dict[str, int],
        errored: bool,
        inputs: Sequence[str],
        outputs: Sequence[str],
    ) -> None:
        """Record this run's decision on the hook's span, keeping the texts in content tags."""
        span = tracing.tracer.current_span()
        if span is None:
            return
        span.set_tags(
            tags={
                "haystack.agent.hook.pii.checked_messages": checked,
                "haystack.agent.hook.pii.triggered": bool(entity_counts) or errored,
                "haystack.agent.hook.pii.action": action,
                "haystack.agent.hook.pii.entities": entity_counts,
                "haystack.agent.hook.pii.error": errored,
            }
        )
        span.set_content_tag("haystack.agent.hook.pii.input", list(inputs))
        span.set_content_tag("haystack.agent.hook.pii.output", list(outputs))
