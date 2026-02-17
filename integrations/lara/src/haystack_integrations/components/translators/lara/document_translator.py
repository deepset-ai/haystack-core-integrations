# SPDX-FileCopyrightText: 2026-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret, deserialize_secrets_inplace
from lara_sdk import Credentials, TextBlock, Translator  # type: ignore[import-untyped]

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
        style: Literal["faithful", "fluid", "creative"] | None = None,
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
            If `None`, the default style is `"faithful"`.
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
            at inference time. Domain adaptation is available on Team and Enterprise plans. You can find more
            detailed information in the
            [Lara documentation](https://developers.laratranslate.com/docs/adapt-to-translation-memories).
        :param glossaries:
            Optional list of glossary IDs. Lara applies these glossaries at inference time to enforce
            consistent terminology (e.g. brand names, product terms, legal or technical phrases) across translations.
            Glossary management and availability depend on your plan.
            You can find more detailed information in the
            [Lara documentation](https://developers.laratranslate.com/docs/manage-glossaries).
        :param reasoning:
            If `True`, uses the Lara Think model for higher-quality translation (multi-step linguistic analysis).
            Increases latency and cost. Available on Pro and Team plans. You can find more detailed information in the
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
            If `None`, the default style is `"faithful"`.
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
            of these memories at inference time. Domain adaptation is available on Team and Enterprise plans.
            You can find more detailed information in the
            [Lara documentation](https://developers.laratranslate.com/docs/adapt-to-translation-memories).
        :param glossaries:
            Optional list of glossary IDs. Lara applies these glossaries at inference time to enforce
            consistent terminology (e.g. brand names, product terms, legal or technical phrases) across translations.
            Glossary management and availability depend on your plan.
            You can find more detailed information in the
            [Lara documentation](https://developers.laratranslate.com/docs/manage-glossaries).
        :param reasoning:
            If `True`, uses the Lara Think model for higher-quality translation (multi-step linguistic analysis).
            Increases latency and cost. Available on Pro and Team plans. You can find more detailed information in the
            [Lara documentation](https://developers.laratranslate.com/docs/translate-text#reasoning-lara-think).
        :return:
            A dictionary with the following keys:
            - `documents`: A list of translated documents.

        :raises ValueError: If any list-valued parameter has length != `len(documents)`.
        """
        if not documents:
            return {"documents": []}

        if self._translator is None:
            credentials = Credentials(
                access_key_id=self.access_key_id.resolve_value(),
                access_key_secret=self.access_key_secret.resolve_value(),
            )
            self._translator = Translator(credentials=credentials)

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
            if cur_doc.content:
                cur_source = validated_params["source_lang"][idx]
                cur_target = validated_params["target_lang"][idx]
                cur_ctx = validated_params["context"][idx]
                cur_instr = validated_params["instructions"][idx]
                cur_style = validated_params["style"][idx]
                cur_adapt = validated_params["adapt_to"][idx]
                cur_gloss = validated_params["glossaries"][idx]
                cur_reason = validated_params["reasoning"][idx]

                text_blocks = [TextBlock(text=cur_doc.content, translatable=True)]
                if cur_ctx is not None:
                    text_blocks.append(TextBlock(text=cur_ctx, translatable=False))

                translation_response = self._translator.translate(
                    text=text_blocks,
                    source=cur_source,
                    target=cur_target,
                    instructions=cur_instr,
                    style=cur_style,
                    adapt_to=cur_adapt,
                    glossaries=cur_gloss,
                    reasoning=cur_reason,
                )
                translation = translation_response.translation[0].text
            else:
                logger.warning(f"Document {cur_doc.id} has no content, skipping translation.")
                translation = cur_doc.content

            meta = {"original_document_id": cur_doc.id, **cur_doc.meta}
            translated_doc = Document(content=translation, meta=meta)
            translated_documents.append(translated_doc)

        return {"documents": translated_documents}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LaraDocumentTranslator":
        """
        Deserializes a LaraDocumentTranslator from a dictionary.

        :param data: Dictionary containing the LaraDocumentTranslator configuration.
        :return: LaraDocumentTranslator instance.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["access_key_id", "access_key_secret"])
        return default_from_dict(cls, data)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes a LaraDocumentTranslator to a dictionary.

        :return: Dictionary containing the LaraDocumentTranslator configuration.
        """
        return default_to_dict(
            self,
            access_key_id=self.access_key_id.to_dict(),
            access_key_secret=self.access_key_secret.to_dict(),
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            context=self.context,
            instructions=self.instructions,
            style=self.style,
            adapt_to=self.adapt_to,
            glossaries=self.glossaries,
            reasoning=self.reasoning,
        )

    @staticmethod
    def _validate_params(
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
        error_msg = "If {param} is a list, it must be the same length as the number of documents."
        validated_params: dict[str, list[Any]] = {}

        if isinstance(source_lang, list) and len(source_lang) != num_documents:
            raise ValueError(error_msg.format(param="source language"))
        validated_params["source_lang"] = (
            [source_lang] * num_documents if not isinstance(source_lang, list) else source_lang
        )

        if isinstance(target_lang, list) and len(target_lang) != num_documents:
            raise ValueError(error_msg.format(param="target language"))
        validated_params["target_lang"] = (
            [target_lang] * num_documents if not isinstance(target_lang, list) else target_lang
        )

        if isinstance(context, list) and len(context) != num_documents:
            raise ValueError(error_msg.format(param="context"))
        validated_params["context"] = [context] * num_documents if not isinstance(context, list) else context

        if isinstance(style, list) and len(style) != num_documents:
            raise ValueError(error_msg.format(param="style"))
        validated_params["style"] = [style] * num_documents if not isinstance(style, list) else style

        if isinstance(instructions, list) and len(instructions) != num_documents:
            raise ValueError(error_msg.format(param="instructions"))
        if instructions is not None:
            validated_params["instructions"] = (
                [[instructions]] * num_documents if not isinstance(instructions, list) else instructions
            )
        else:
            validated_params["instructions"] = [None] * num_documents

        if isinstance(reasoning, list) and len(reasoning) != num_documents:
            raise ValueError(error_msg.format(param="reasoning"))
        validated_params["reasoning"] = [reasoning] * num_documents if not isinstance(reasoning, list) else reasoning

        if (
            isinstance(adapt_to, list)
            and len(adapt_to) > 0
            and isinstance(adapt_to[0], list)
            and len(adapt_to) != num_documents
        ):
            error_msg = "If adapt to is a list of lists, it must be the same length as the number of documents."
            raise ValueError(error_msg)
        validated_params["adapt_to"] = (
            [adapt_to] * num_documents
            if not (isinstance(adapt_to, list) and len(adapt_to) > 0 and isinstance(adapt_to[0], list))
            else adapt_to
        )

        if (
            isinstance(glossaries, list)
            and len(glossaries) > 0
            and isinstance(glossaries[0], list)
            and len(glossaries) != num_documents
        ):
            error_msg = "If glossaries is a list of lists, it must be the same length as the number of documents."
            raise ValueError(error_msg)
        validated_params["glossaries"] = (
            [glossaries] * num_documents
            if not (isinstance(glossaries, list) and len(glossaries) > 0 and isinstance(glossaries[0], list))
            else glossaries
        )
        return validated_params
