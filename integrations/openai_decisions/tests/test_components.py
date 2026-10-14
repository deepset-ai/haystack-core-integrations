# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document, Pipeline
from haystack.core.serialization import component_from_dict

from haystack_integrations.components.classifiers.openai_decisions import OpenAIDecisionsDocumentClassifier
from haystack_integrations.components.routers.openai_decisions import OpenAIDecisionsTextRouter

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "choices": [{"value": "billing"}, {"value": "technical"}],
    },
    "damaged": {"type": "predicate", "instructions": "Is the item damaged?"},
    "severity": {
        "type": "score",
        "name": "severity",
        "instructions": "Rate severity",
        "levels": [{"label": "low"}, {"label": "high"}],
    },
}


def _answer(name, type_, **kwargs):
    answer = MagicMock()
    answer.name = name
    answer.type = type_
    answer.confidence = kwargs.get("confidence", 0.9)
    answer.choice = kwargs.get("choice", "billing")
    answer.model_dump.return_value = {
        "type": type_,
        "name": name,
        **kwargs,
    }
    return answer


@pytest.fixture(autouse=True)
def api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def test_classifier_validates_questions():
    with pytest.raises(ValueError, match="at least one question"):
        OpenAIDecisionsDocumentClassifier(questions={})
    with pytest.raises(ValueError, match="needs `choices`"):
        OpenAIDecisionsDocumentClassifier(questions={"q": {"type": "choice", "instructions": "Choose"}})


def test_classifier_serialization_round_trip():
    classifier = OpenAIDecisionsDocumentClassifier(questions=QUESTIONS, model="gpt-6-luna", metadata_field="answers")
    payload = classifier.to_dict()
    restored = component_from_dict(OpenAIDecisionsDocumentClassifier, payload, name="classifier")
    assert restored.questions == QUESTIONS
    assert restored.metadata_field == "answers"
    pipeline = Pipeline()
    pipeline.add_component("classifier", classifier)
    assert Pipeline.loads(pipeline.dumps()).get_component("classifier").to_dict() == payload


def test_classifier_calls_decisions_and_stores_answers():
    classifier = OpenAIDecisionsDocumentClassifier(questions=QUESTIONS)
    decision = MagicMock(
        answers=[_answer("department", "choice", choice="billing"), _answer("damaged", "predicate", probability=0.1)]
    )
    with patch(
        "haystack_integrations.components.classifiers.openai_decisions.document_classifier.OpenAI"
    ) as mock_openai:
        client = mock_openai.return_value
        client.decisions.create.return_value = decision
        result = classifier.run([Document(content="I was charged twice")])
    assert result["failed_documents"] == []
    client.decisions.create.assert_called_once()
    sent_questions = client.decisions.create.call_args.kwargs["questions"]
    assert [question["name"] for question in sent_questions] == list(QUESTIONS)
    assert result["documents"][0].meta["openai_decisions"]["department"]["choice"] == "billing"


def test_classifier_separates_failed_documents():
    classifier = OpenAIDecisionsDocumentClassifier(questions=QUESTIONS)
    with patch(
        "haystack_integrations.components.classifiers.openai_decisions.document_classifier.OpenAI"
    ) as mock_openai:
        mock_openai.return_value.decisions.create.side_effect = RuntimeError("offline")
        result = classifier.run([Document(content="hello")])
    assert result["documents"] == []
    assert result["failed_documents"][0].meta["classification_error"] == "offline"


def test_router_routes_choice():
    router = OpenAIDecisionsTextRouter(labels={"billing": "Invoices", "technical": "Bugs"})
    answer = _answer("route", "choice", choice="billing")
    with patch("haystack_integrations.components.routers.openai_decisions.text_router.OpenAI") as mock_openai:
        mock_openai.return_value.decisions.create.return_value.answers = [answer]
        assert router.run("charged twice") == {"billing": "charged twice"}


def test_router_low_confidence_fallback():
    router = OpenAIDecisionsTextRouter(labels=["billing", "technical"], min_confidence=0.7)
    answer = _answer("route", "choice", choice="billing", confidence=0.2)
    with patch("haystack_integrations.components.routers.openai_decisions.text_router.OpenAI") as mock_openai:
        mock_openai.return_value.decisions.create.return_value.answers = [answer]
        assert router.run("charged twice") == {"low_confidence": "charged twice"}


def test_router_rejects_invalid_input_and_labels():
    with pytest.raises(TypeError, match="expects a string"):
        OpenAIDecisionsTextRouter(labels=["billing"]).run(123)
    with pytest.raises(ValueError, match="reserved"):
        OpenAIDecisionsTextRouter(labels=["low_confidence", "billing"])
