# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

"""
The tools the agents use.

- `web_search`: a `ComponentTool` wrapping the Tavily Haystack integration.
- `read_url`: a `PipelineTool` wrapping a small summarization pipeline. The fetched
  page is routed by MIME type (`FileTypeRouter`) to `HTMLToDocument` or
  `PyPDFToDocument` so PDFs are parsed too, then summarized toward the question.
- `think_tool`: the reflection no-op — a plain `@tool`.
"""

from typing import Annotated, Any

from haystack import Document, Pipeline, component, logging
from haystack.components.builders import ChatPromptBuilder
from haystack.components.converters import HTMLToDocument, PyPDFToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.generators.chat.types import ChatGenerator
from haystack.components.joiners import DocumentJoiner
from haystack.components.routers import FileTypeRouter
from haystack.core.serialization import generate_qualified_class_name
from haystack.lazy_imports import LazyImport
from haystack.tools import ComponentTool, PipelineTool, tool

from haystack_integrations.components.agents.agent_pack.deep_research import prompts

with LazyImport(message="Run 'pip install tavily-haystack'") as tavily_import:
    from haystack_integrations.components.websearch.tavily import TavilyWebSearch

logger = logging.getLogger(__name__)


def _format_search_results(documents: list[Document]) -> str:
    """Format `web_search` results as the tool-result string: title + exact URL + snippet."""
    if not documents:
        return "No results."
    blocks = []
    for d in documents:
        url = d.meta.get("url", "")
        title = d.meta.get("title", "Untitled")
        snippet = (d.content or "").strip()
        blocks.append(f"- {title}\n  URL: {url}\n  {snippet}")
    logger.info(
        "web_search -> {count} results: {urls}", count=len(documents), urls=[d.meta.get("url") for d in documents]
    )
    return "\n".join(blocks)


class TavilyWebSearchTool(ComponentTool):
    """
    `web_search` tool: a ComponentTool over TavilyWebSearch (async-capable).

    :param top_k: Results returned per `web_search` call.
    """

    def __init__(self, top_k: int = 10) -> None:
        tavily_import.check()
        self._top_k = top_k
        super().__init__(
            component=TavilyWebSearch(top_k=top_k),
            name="web_search",
            description=(
                "Search the web. Returns the top results, each with title, exact URL and a content "
                "snippet. Cite the exact URLs verbatim."
            ),
            outputs_to_string={"source": "documents", "handler": _format_search_results},
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the tool to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return {"type": generate_qualified_class_name(type(self)), "data": {"top_k": self._top_k}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TavilyWebSearchTool":
        """
        Deserialize the tool from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized tool.
        """
        return cls(top_k=data["data"]["top_k"])


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
