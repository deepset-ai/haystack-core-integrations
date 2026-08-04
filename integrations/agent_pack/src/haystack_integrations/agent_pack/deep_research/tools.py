# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
The tools defined here for the agents.

- `read_url`: a `PipelineTool` wrapping a small summarization pipeline. The fetched
  page is routed by MIME type (`FileTypeRouter`) to `HTMLToDocument` or
  `PyPDFToDocument` so PDFs are parsed too, then summarized toward the question.
- `think_tool`: the reflection no-op — a plain `@tool`.

`web_search` is the `TavilyWebSearchTool` provided by the Tavily integration; it is wired in `agent.py`.
"""

from typing import Annotated

from haystack import Document, Pipeline, component, logging
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import HTMLToDocument, PyPDFToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.generators.chat.types import ChatGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.routers import FileTypeRouter
from haystack.tools import PipelineTool, tool

from haystack_integrations.agent_pack.deep_research import prompts

logger = logging.getLogger(__name__)


@component
class ContentGate:
    """Content gate for `read_url`: pass documents to the summarizer only when the page actually has text."""

    @component.output_types(documents=list[Document], empty=bool)
    def run(self, documents: list[Document]) -> dict:
        """
        Forward documents to the summarizer only when the fetched page has readable text.

        :param documents: Documents produced from the fetched page.
        :returns: `{"documents": [...]}` when the page has text, otherwise `{"empty": True}`.
        """
        if documents and documents[0].content:
            return {"documents": documents}
        return {"empty": True}


def _read_url_result(result: dict) -> str:
    """
    Turn the `read_url` pipeline result into one line for the agent (an `outputs_to_string` handler).

    We don't treat HTTP/network failures here: the `LinkContentFetcher` has `raise_on_failure=True`, so these
    errors are sent to the agent wrapped in an error tool-result message.
    """
    replies = result.get("replies")
    if replies and replies[0].text:
        return replies[0].text

    unsupported = result.get("unclassified")
    if unsupported:
        content_type = unsupported[0].meta.get("content_type", "unknown")
        return f"Page not read: unsupported content type {content_type}; only HTML and PDF are read."

    return "Page not read: the page returned no readable text."


def make_read_url_tool(*, summarizer_llm: ChatGenerator, max_content_length: int) -> PipelineTool:
    """
    `read_url` = PipelineTool over a fetch -> convert to text -> summarize-based-on-question pipeline.

    :param summarizer_llm: LLM that summarizes the fetched page toward the agent's question.
    :param max_content_length: Raw page chars fed to the summarizer (pre-summarization cap).
    """
    template = prompts.SUMMARIZE_TEMPLATE.replace("__MAXLEN__", str(max_content_length))

    pipe = Pipeline()
    pipe.add_component("fetcher", LinkContentFetcher(raise_on_failure=True, retry_attempts=2, timeout=10))
    pipe.add_component("router", FileTypeRouter(mime_types=["text/html", "application/pdf"]))
    pipe.add_component("html", HTMLToDocument())
    pipe.add_component("pdf", PyPDFToDocument())
    pipe.add_component("joiner", DocumentJoiner())
    pipe.add_component("content_gate", ContentGate())
    pipe.add_component("builder", ChatPromptBuilder(template=template, required_variables=["question", "documents"]))
    pipe.add_component("summarizer", summarizer_llm)
    pipe.connect("fetcher.streams", "router.sources")
    pipe.connect("router.text/html", "html.sources")
    pipe.connect("router.application/pdf", "pdf.sources")
    pipe.connect("html.documents", "joiner.documents")
    pipe.connect("pdf.documents", "joiner.documents")
    pipe.connect("joiner.documents", "content_gate.documents")
    pipe.connect("content_gate.documents", "builder.documents")
    pipe.connect("builder.prompt", "summarizer.messages")

    return PipelineTool(
        pipeline=pipe,
        name="read_url",
        description=(
            "Fetch a web page or PDF and get a summary focused on your question. Provide the URL to "
            "read and the specific question you want answered from it. Use when a snippet is too shallow."
        ),
        input_mapping={"urls": ["fetcher.urls"], "question": ["builder.question"]},
        outputs_to_string={"handler": _read_url_result},
    )


@tool
def think_tool(
    reflection: Annotated[str, "What you learned, what is still missing, and whether to keep searching or stop."],
) -> str:
    """Record a brief reflection to plan your research. Call this between searches."""
    logger.info("think: {reflection}", reflection=reflection)
    return "Reflection noted"
