import logging

from haystack import Document, Pipeline
from haystack.components.generators.chat import MockChatGenerator
from haystack.components.rankers import LLMRanker

from haystack_integrations.evaluation.harness_log_collector import (
    MAX_DISTINCT_MESSAGES,
    MAX_MESSAGE_CHARS,
    HarnessLogCollector,
)


class TestHarnessLogCollector:
    def test_repeated_messages(self):
        with HarnessLogCollector().collect() as logs:
            for _ in range(20):
                logging.getLogger("haystack.components.query.query_expander").warning("Generated 4 but 1 requested.")
        assert logs.to_list() == [
            {
                "level": "WARNING",
                "logger": "haystack.components.query.query_expander",
                "message": "Generated 4 but 1 requested.",
                "count": 20,
            }
        ]

    def test_ignores_records_below_warning(self):
        with HarnessLogCollector().collect() as logs:
            logging.getLogger("haystack").info("routine progress")
            logging.getLogger("haystack").error("something broke")
        assert [entry["level"] for entry in logs.to_list()] == ["ERROR"]

    def test_caps_distinct_messages(self):
        with HarnessLogCollector().collect() as logs:
            for index in range(MAX_DISTINCT_MESSAGES + 5):
                logging.getLogger("haystack").warning("distinct %s", index)
        assert len(logs.to_list()) == MAX_DISTINCT_MESSAGES
        assert logs.dropped == 5

    def test_truncates_long_message(self):
        with HarnessLogCollector().collect() as logs:
            logging.getLogger("haystack").warning("x" * (MAX_MESSAGE_CHARS + 500))
        assert len(logs.to_list()[0]["message"]) == MAX_MESSAGE_CHARS

    def test_restores_logging(self):
        logger = logging.getLogger("haystack")
        before_handlers, before_level = list(logger.handlers), logger.level
        with HarnessLogCollector().collect() as logs:
            logger.warning("during")
        logger.warning("after")
        assert [entry["message"] for entry in logs.to_list()] == ["during"]
        assert logger.handlers == before_handlers
        assert logger.level == before_level

    def test_ignores_other_loggers(self):
        with HarnessLogCollector(logger_names=("haystack",)).collect() as logs:
            logging.getLogger("haystack.components.rankers").warning("mine")
            logging.getLogger("someone_else").warning("not mine")
        assert [entry["message"] for entry in logs.to_list()] == ["mine"]

    def test_collect_on_pipeline(self):
        """
        The exact shape that once scored as an improvement: `LLMRanker` catches a rejected parameter, returns the
        documents in the order it received them, and reports it only to a logger. The run succeeds, so nothing but
        the warning distinguishes an unranked answer from a ranked one.
        """

        def reject(*_args, **_kwargs):
            message = "Error code: 400 - temperature does not support 0.0 with this model"
            raise RuntimeError(message)

        documents = [Document(content="first"), Document(content="second"), Document(content="third")]
        pipeline = Pipeline()
        # raise_on_failure is False by default, which is what makes the failure invisible to the score.
        pipeline.add_component("ranker", LLMRanker(chat_generator=MockChatGenerator(response_fn=reject), top_k=1))
        with HarnessLogCollector().collect() as logs:
            result = pipeline.run({"ranker": {"query": "which one", "documents": documents}})
        reported = logs.to_list()
        assert any("LLMRanker failed during chat generation" in entry["message"] for entry in reported)
        assert any("temperature does not support 0.0" in entry["message"] for entry in reported)
        # The run completed and handed back its input untouched, so every other metric looks ordinary.
        assert [document.content for document in result["ranker"]["documents"]] == ["first", "second", "third"]
