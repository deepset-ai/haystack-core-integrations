from haystack_integrations.evaluation.dataclasses import ModelTokenUsage
from haystack_integrations.tracing.agent_pack import ReportedUsage, SpanRecord
from haystack_integrations.tracing.agent_pack.span_records import _eval_case_summary_from_records


def generator_record(model="m", tokens=None, **overrides):
    """One span record shaped like a model call that reported a single reply."""
    return SpanRecord(
        is_generator_span=True,
        reported_output=True,
        reported_usage=[
            ReportedUsage(model=model, tokens=tokens if tokens is not None else {"input_tokens": 3, "output_tokens": 1})
        ],
        **overrides,
    )


class TestEvalCaseSummaryFromRecords:
    def test_sums_repeated_models(self):
        """Folding is a pure function of the records, so it can be checked without running anything."""
        summary = _eval_case_summary_from_records(records=[generator_record(), generator_record()])
        assert summary.llm_calls == 2
        assert summary.models["m"] == ModelTokenUsage(input_tokens=6, output_tokens=2)
        assert summary.all_tokens_reported

    def test_call_without_output(self):
        """A generator span that never emitted an output tag spent tokens nobody can account for."""
        silent = SpanRecord(is_generator_span=True)
        summary = _eval_case_summary_from_records(records=[generator_record(), silent])
        assert summary.all_tokens_reported is False
        # The silent call is not counted, because nothing says it produced anything.
        assert summary.llm_calls == 1

    def test_unquantified_usage(self):
        replied_without_usage = generator_record(model="m", tokens={})
        replied_without_model = generator_record(model=None)
        for record in (replied_without_usage, replied_without_model):
            summary = _eval_case_summary_from_records(records=[record])
            assert summary.all_tokens_reported is False
            assert summary.models == {}
            # The call still happened, so it is still counted.
            assert summary.llm_calls == 1

    def test_generator_output_not_a_stage(self):
        """A reply count says nothing about how much reached the next stage, and would collide with its owner."""
        ranker = SpanRecord(component_name="ranker", output_sizes={"documents": 4})
        nested = generator_record(component_name="ranker", output_sizes={"replies": 1})
        summary = _eval_case_summary_from_records(records=[nested, ranker])
        assert summary.outputs == {"ranker": {"documents": 4}}
