import logging

from haystack_integrations.agent_pack.evaluation.component_logs import (
    MAX_DISTINCT_MESSAGES,
    MAX_MESSAGE_CHARS,
    ComponentLogCollector,
)


def test_a_swallowed_component_failure_is_reported_with_its_remedy():
    """`LLMRanker` returns documents unranked and says so only in a warning; that text names the fix."""
    with ComponentLogCollector().collect() as logs:
        logging.getLogger("haystack.components.rankers.llm_ranker").warning(
            "LLMRanker failed during chat generation. Returning fallback order. Error: temperature is unsupported"
        )

    assert logs.to_list() == [
        {
            "level": "WARNING",
            "logger": "haystack.components.rankers.llm_ranker",
            "message": "LLMRanker failed during chat generation. Returning fallback order. "
            "Error: temperature is unsupported",
            "count": 1,
        }
    ]


def test_a_warning_repeated_across_cases_is_counted_once():
    with ComponentLogCollector().collect() as logs:
        for _ in range(20):
            logging.getLogger("haystack.components.query.query_expander").warning("Generated 4 but 1 requested.")

    assert len(logs.to_list()) == 1
    assert logs.to_list()[0]["count"] == 20


def test_information_below_warning_is_not_collected():
    with ComponentLogCollector().collect() as logs:
        logging.getLogger("haystack").info("routine progress")
        logging.getLogger("haystack").error("something broke")

    assert [entry["level"] for entry in logs.to_list()] == ["ERROR"]


def test_distinct_messages_are_capped_and_the_overflow_is_counted():
    with ComponentLogCollector().collect() as logs:
        for index in range(MAX_DISTINCT_MESSAGES + 5):
            logging.getLogger("haystack").warning("distinct %s", index)

    assert len(logs.to_list()) == MAX_DISTINCT_MESSAGES
    assert logs.dropped == 5


def test_a_long_message_is_truncated():
    with ComponentLogCollector().collect() as logs:
        logging.getLogger("haystack").warning("x" * (MAX_MESSAGE_CHARS + 500))

    assert len(logs.to_list()[0]["message"]) == MAX_MESSAGE_CHARS


def test_collection_does_not_outlive_the_evaluation_or_disturb_existing_logging():
    logger = logging.getLogger("haystack")
    before_handlers, before_level = list(logger.handlers), logger.level

    with ComponentLogCollector().collect() as logs:
        logger.warning("during")
    logger.warning("after")

    assert [entry["message"] for entry in logs.to_list()] == ["during"]
    assert logger.handlers == before_handlers
    assert logger.level == before_level


def test_only_the_configured_trees_are_listened_to():
    with ComponentLogCollector(logger_names=("haystack",)).collect() as logs:
        logging.getLogger("haystack.components.rankers").warning("mine")
        logging.getLogger("someone_else").warning("not mine")

    assert [entry["message"] for entry in logs.to_list()] == ["mine"]
