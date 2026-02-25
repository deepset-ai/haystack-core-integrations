# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from haystack import Document, component, logging
from haystack.utils import Secret
from lara_sdk import Credentials, TextBlock, Translator

logger = logging.getLogger(__name__)


@component
class LaraDocumentTranslator:
    """
    Translates the text content of Haystack Documents using translated's Lara translation API.

    Lara is an adaptive translation AI that combines the fluency and context handling
    of LLMs with low hallucination and latency. It adapts to domains at inference time
    using optional context, instructions, translation memories, and glossaries. You can find
    more detailed information in the [Lara documentation](https://developers.laratranslate.com/docs/introduction).


    ### Usage example
    ```python
    from haystack import Document
    from haystack.utils import Secret
    from haystack_integrations.components.lara import LaraDocumentTranslator

    translator = LaraDocumentTranslator(
        access_key_id=Secret.from_env_var("LARA_ACCESS_KEY_ID"),
        access_key_secret=Secret.from_env_var("LARA_ACCESS_KEY_SECRET"),
        source_lang="en-US",
        target_lang="de-DE",
    )

    doc = Document(content="Hello, world!")
    result = translator.run(documents=[doc])
    print(result["documents"][0].content)
    ```
    """

    def __init__(
        self,
        access_key_id: Secret = Secret.from_env_var("LARA_ACCESS_KEY_ID"),
        access_key_secret: Secret = Secret.from_env_var("LARA_ACCESS_KEY_SECRET"),
        source_lang: str | None = None,
        target_lang: str | None = None,
        context: str | None = None,
        instructions: str | None = None,
        style: Literal["faithful", "fluid", "creative"] = "faithful",
        adapt_to: list[str] | None = None,
        glossaries: list[str] | None = None,
        reasoning: bool = False,
    ):
        """
        Creats an instance of the LaraDocumentTranslator component.

        :param access_key_id:
            Lara API access key ID. Defaults to the `LARA_ACCESS_KEY_ID` environment variable.
        :param access_key_secret:
            Lara API access key secret. Defaults to the `LARA_ACCESS_KEY_SECRET` environment variable.
        :param source_lang:
            Language code of the source text. If `None`, Lara auto-detects the source language.
            Use locale codes from the
            [supported languages list](https://developers.laratranslate.com/docs/supported-languages).
        :param target_lang:
            Language code of the target text.
            Use locale codes from the
            [supported languages list](https://developers.laratranslate.com/docs/supported-languages).
        :param context:
            Optional external context: text that is not translated but is sent to Lara to
            improve translation quality (e.g. surrounding sentences, prior messages).
            You can find more detailed information in the
            [Lara documentation](https://developers.laratranslate.com/docs/adapt-to-context).
        :param instructions:
            Optional natural-language instructions to guide translation and
            specify domain-specific terminology (e.g. "Be formal", "Use a professional tone").
            You can find more detailed information in the
            [Lara documentation](https://developers.laratranslate.com/docs/adapt-to-instructions).
        :param style:
            One of `"faithful"`, `"fluid"`, or `"creative"`.
            Default is `"faithful"`.
            Style description:
            - `"faithful"`: For accuracy and precision. Keeps original structure and meaning.
                Ideal for manuals, legal documents.
            - `"fluid"`: For readability and natural flow. Smooth, conversational. Good for general content.
            - `"creative"`: For artistic and creative expression. Best for literature, marketing, or content
                where impact and tone matter more than literal wording.
            You can find more detailed information in the
            [Lara documentation](https://support.laratranslate.com/en/translation-styles).
        :param adapt_to:
            Optional list of translation memory IDs. Lara adapts to the style and terminology of these memories
            at inference time. Domain adaptation is available depending on your plan. You can find more
            detailed information in the
            [Lara documentation](https://developers.laratranslate.com/docs/adapt-to-translation-memories).
        :param glossaries:
            Optional list of glossary IDs. Lara applies these glossaries at inference time to enforce
            consistent terminology (e.g. brand names, product terms, legal or technical phrases) across translations.
            Glossary management and availability depends on your plan.
            You can find more detailed information in the
            [Lara documentation](https://developers.laratranslate.com/docs/manage-glossaries).
        :param reasoning:
            If `True`, uses the Lara Think model for higher-quality translation (multi-step linguistic analysis).
            Increases latency and cost. Availability depends on your plan. You can find more detailed information in the
            [Lara documentation](https://developers.laratranslate.com/docs/translate-text#reasoning-lara-think).
        """

        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.context = context
        self.instructions = instructions
        self.style = style
        self.adapt_to = adapt_to
        self.glossaries = glossaries
        self.reasoning = reasoning
        self._translator: Translator | None = None

    def warm_up(self) -> None:
        """
        Warm up the Lara translator by initializing the client.
        """
        if self._translator is None:
            credentials = Credentials(
                access_key_id=self.access_key_id.resolve_value(),
                access_key_secret=self.access_key_secret.resolve_value(),
            )
            self._translator = Translator(credentials=credentials)

    @component.output_types(documents=list[Document])
    def run(
        self,
        documents: list[Document],
        source_lang: str | list[str | None] | None = None,
        target_lang: str | list[str] | None = None,
        context: str | list[str] | None = None,
        instructions: str | list[str] | None = None,
        style: str | list[str] | None = None,
        adapt_to: list[str] | list[list[str]] | None = None,
        glossaries: list[str] | list[list[str]] | None = None,
        reasoning: bool | list[bool] | None = None,
    ) -> dict[str, list[Document]]:
        """
        Translate the text content of each input Document using the Lara API.

        Any of the translation parameters (source_lang, target_lang, context,
        instructions, style, adapt_to, glossaries, reasoning) can be passed here
        to override the defaults set when creating the component. They can be a single value
        (applied to all documents) or a list of values with the same length as
        `documents` for per-document settings.

        :param documents:
            List of Haystack Documents whose `content` is to be translated.
        :param source_lang:
            Source language code(s). Use locale codes from the
            [supported languages list](https://developers.laratranslate.com/docs/supported-languages).
            If `None`, Lara auto-detects the source language. Single value or list (one per document).
        :param target_lang:
            Target language code(s). Use locale codes from the
            [supported languages list](https://developers.laratranslate.com/docs/supported-languages).
            Single value or list (one per document).
        :param context:
            Optional external context: text that is not translated but is sent to Lara to
            improve translation quality (e.g. surrounding sentences, prior messages).
            You can find more detailed information in the
            [Lara documentation](https://developers.laratranslate.com/docs/adapt-to-context).
        :param instructions:
            Optional natural-language instructions to guide translation and specify
            domain-specific terminology (e.g. "Be formal", "Use a professional tone").
            You can find more detailed information in the
            [Lara documentation](https://developers.laratranslate.com/docs/adapt-to-instructions).
        :param style:
            One of `"faithful"`, `"fluid"`, or `"creative"`.
            Style description:
            - `"faithful"`: For accuracy and precision. Keeps original structure and meaning.
                Ideal for manuals, legal documents.
            - `"fluid"`: For readability and natural flow. Smooth, conversational. Good for general content.
            - `"creative"`: For artistic and creative expression. Best for literature, marketing, or content
                where impact and tone matter more than literal wording.
            You can find more detailed information in the
            [Lara documentation](https://support.laratranslate.com/en/translation-styles).
        :param adapt_to:
            Optional list of translation memory IDs. Lara adapts to the style and terminology
            of these memories at inference time. Domain adaptation is available depending on your plan.
            You can find more detailed information in the
            [Lara documentation](https://developers.laratranslate.com/docs/adapt-to-translation-memories).
        :param glossaries:
            Optional list of glossary IDs. Lara applies these glossaries at inference time to enforce
            consistent terminology (e.g. brand names, product terms, legal or technical phrases) across translations.
            Glossary management and availability depends on your plan.
            You can find more detailed information in the
            [Lara documentation](https://developers.laratranslate.com/docs/manage-glossaries).
        :param reasoning:
            If `True`, uses the Lara Think model for higher-quality translation (multi-step linguistic analysis).
            Increases latency and cost. Availability depends on your plan. You can find more detailed information in the
            [Lara documentation](https://developers.laratranslate.com/docs/translate-text#reasoning-lara-think).
        :return:
            A dictionary with the following keys:
            - `documents`: A list of translated documents.

        :raises ValueError: If any list-valued parameter has length != `len(documents)`.
        """
        if not documents:
            return {"documents": []}

        if self._translator is None:
            self.warm_up()

        source_lang = source_lang or self.source_lang
        target_lang = target_lang or self.target_lang
        context = context or self.context
        instructions = instructions or self.instructions
        style = style or self.style
        adapt_to = adapt_to or self.adapt_to
        glossaries = glossaries or self.glossaries
        reasoning = reasoning or self.reasoning

        validated_params = self._validate_params(
            len(documents),
            source_lang,
            target_lang,
            context,
            style,
            instructions,
            adapt_to,
            glossaries,
            reasoning,
        )

        translated_documents = []
        for idx, cur_doc in enumerate(documents):
            if not cur_doc.content:
                logger.warning(f"Document {cur_doc.id} has no content, skipping translation.")
                translation = cur_doc.content
            else:
                params = {key: value[idx] for key, value in validated_params.items()}
                text_blocks = [TextBlock(text=cur_doc.content, translatable=True)]
                if params["context"] is not None:
                    text_blocks.append(TextBlock(text=params["context"], translatable=False))

                # Ignoring type because the lara client is only initialized after warm_up and None before
                translation_response = self._translator.translate(  # type: ignore[union-attr]
                    text=text_blocks,
                    source=params["source_lang"],
                    target=params["target_lang"],
                    instructions=params["instructions"],
                    style=params["style"],
                    adapt_to=params["adapt_to"],
                    glossaries=params["glossaries"],
                    reasoning=params["reasoning"],
                )
                translation = translation_response.translation[0].text

            meta = {"original_document_id": cur_doc.id, **cur_doc.meta}
            translated_doc = Document(content=translation, meta=meta)
            translated_documents.append(translated_doc)

        return {"documents": translated_documents}

    @staticmethod
    def _normalize_parameter(
        name: str,
        value: Any,
        num_documents: int,
        allow_list_of_lists: bool = False,
        wrap_scalar: bool = False,
    ) -> list[Any]:
        """
        Validate a single translation parameter and broadcast it to a per-document list.

        :param name: Human-readable parameter name used in error messages.
        :param value: The raw parameter value (scalar, list, or list-of-lists).
        :param num_documents: Expected number of documents.
        :param allow_list_of_lists: If True, treat a list whose first element is
            itself a list as already per-document (used for adapt_to / glossaries).
        :param wrap_scalar: If True and value is a non-None scalar, wrap it in a
            single-element list before replicating (used for instructions).
        :return: A list of length `num_documents`.
        :raises ValueError: If a list-valued parameter has the wrong length.
        """
        if allow_list_of_lists:
            is_per_doc = isinstance(value, list) and len(value) > 0 and isinstance(value[0], list)
            if is_per_doc:
                if len(value) != num_documents:
                    msg = f"If {name} is a list of lists, it must be the same length as the number of documents."
                    raise ValueError(msg)
                return value
            return [value] * num_documents

        if isinstance(value, list):
            if len(value) != num_documents:
                msg = f"If {name} is a list, it must be the same length as the number of documents."
                raise ValueError(msg)
            return value

        if wrap_scalar and value is not None:
            value = [value]
        return [value] * num_documents

    def _validate_params(
        self,
        num_documents: int,
        source_lang: str | list[str | None] | None,
        target_lang: str | list[str] | None,
        context: str | list[str] | None,
        style: str | list[str] | None,
        instructions: str | list[str] | None,
        adapt_to: list[str] | list[list[str]] | None,
        glossaries: list[str] | list[list[str]] | None,
        reasoning: bool | list[bool] | None,
    ) -> dict[str, list[Any]]:
        """
        Validates translation parameters and normalizes them to per-document lists.

        :param num_documents: Number of documents in the current batch.
        :param source_lang: Source language (scalar or list of length num_documents).
        :param target_lang: Target language (scalar or list of length num_documents).
        :param context: Context (scalar or list of length num_documents).
        :param style: Style (scalar or list of length num_documents).
        :param instructions: Instructions (scalar or list of length num_documents).
        :param adapt_to: Adaptation memory IDs (list or list of lists, one per doc).
        :param glossaries: Glossary IDs (list or list of lists, one per doc).
        :param reasoning: Reasoning flag (scalar or list of length num_documents).
        :return: Dictionary mapping each parameter name to a list of length
            `num_documents`.
        :raises ValueError: If any list-valued parameter has length != num_documents.
        """
        return {
            "source_lang": self._normalize_parameter("source language", source_lang, num_documents),
            "target_lang": self._normalize_parameter("target language", target_lang, num_documents),
            "context": self._normalize_parameter("context", context, num_documents),
            "style": self._normalize_parameter("style", style, num_documents),
            "instructions": self._normalize_parameter("instructions", instructions, num_documents, wrap_scalar=True),
            "reasoning": self._normalize_parameter("reasoning", reasoning, num_documents),
            "adapt_to": self._normalize_parameter("adapt to", adapt_to, num_documents, allow_list_of_lists=True),
            "glossaries": self._normalize_parameter("glossaries", glossaries, num_documents, allow_list_of_lists=True),
        }
