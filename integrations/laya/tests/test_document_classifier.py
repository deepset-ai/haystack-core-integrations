# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from haystack import Document, Pipeline
from haystack.utils import ComponentDevice, Secret

from haystack_integrations.components.classifiers.laya import LayaDocumentClassifier

COMPONENT_TYPE = "haystack_integrations.components.classifiers.laya.document_classifier.LayaDocumentClassifier"

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this ticket?",
        "criteria": ["billing", "technical"],
    },
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
}


def _result(choice: str) -> dict:
    return {
        "model": "laya-rl-agent",
        "answers": {
            "department": {"type": "choice", "choice": choice, "probabilities": {}, "confidence": 0.9},
            "refund": {"type": "noul", "noul": 0.8},
        },
        "usage": {"input_tokens": 10, "output_tokens": 0},
    }


class TestInit:
    def test_init_default(self):
        component = LayaDocumentClassifier(questions=QUESTIONS)
        assert component.questions == QUESTIONS
        assert component.model == "convaiinnovations/laya"
        assert component.subfolder is None
        assert component.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert component.metadata_field == "laya"
        assert component._agent is None

    @pytest.mark.parametrize(
        ("questions", "match"),
        [
            ({}, "at least one question"),
            ({"q": {"type": "yesno", "instructions": "?"}}, "has type 'yesno'"),
            ({"q": {"type": "noul"}}, "no `instructions`"),
        ],
    )
    def test_init_invalid_questions(self, questions, match):
        with pytest.raises(ValueError, match=match):
            LayaDocumentClassifier(questions=questions)


class TestSerialization:
    def test_to_dict_from_dict_round_trip(self, monkeypatch):
        monkeypatch.setenv("LAYA_HF_TOKEN", "test-token")
        component = LayaDocumentClassifier(
            questions=QUESTIONS,
            model="my-org/laya-finetune",
            subfolder="multilingual",
            revision="abc123",
            device=ComponentDevice.from_str("cpu"),
            token=Secret.from_env_var("LAYA_HF_TOKEN"),
            classification_field="summary",
            metadata_field="decisions",
            batch_size=8,
            min_confidence=0.5,
        )
        data = component.to_dict()
        assert data == {
            "type": COMPONENT_TYPE,
            "init_parameters": {
                "questions": QUESTIONS,
                "model": "my-org/laya-finetune",
                "subfolder": "multilingual",
                "revision": "abc123",
                "device": ComponentDevice.from_str("cpu").to_dict(),
                "token": {"env_vars": ["LAYA_HF_TOKEN"], "strict": True, "type": "env_var"},
                "classification_field": "summary",
                "metadata_field": "decisions",
                "batch_size": 8,
                "min_confidence": 0.5,
            },
        }
        restored = LayaDocumentClassifier.from_dict(data)
        assert restored.questions == QUESTIONS
        assert restored.model == "my-org/laya-finetune"
        assert restored.subfolder == "multilingual"
        assert restored.revision == "abc123"
        assert restored.device == ComponentDevice.from_str("cpu")
        assert restored.token.resolve_value() == "test-token"
        assert restored.classification_field == "summary"
        assert restored.metadata_field == "decisions"
        assert restored.batch_size == 8
        assert restored.min_confidence == 0.5

    def test_pipeline_round_trip(self):
        pipeline = Pipeline()
        pipeline.add_component("classifier", LayaDocumentClassifier(questions=QUESTIONS))
        restored = Pipeline.loads(pipeline.dumps())
        assert restored.get_component("classifier").to_dict() == pipeline.get_component("classifier").to_dict()


class TestWarmUp:
    @patch("haystack_integrations.components.classifiers.laya.document_classifier.load")
    def test_warm_up(self, mock_load, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "test-token")
        component = LayaDocumentClassifier(
            questions=QUESTIONS, subfolder="multilingual", revision="abc123", device=ComponentDevice.from_str("cpu")
        )
        component.warm_up()
        component.warm_up()
        mock_load.assert_called_once_with(
            "convaiinnovations/laya", device="cpu", token="test-token", subfolder="multilingual", revision="abc123"
        )
        assert component._agent is mock_load.return_value


class TestRun:
    def test_run(self):
        component = LayaDocumentClassifier(questions=QUESTIONS, batch_size=4, min_confidence=0.5)
        component._agent = MagicMock()
        component._agent.predict_batch.return_value = [_result("billing"), _result("technical")]
        documents = [
            Document(content="I was charged twice.", meta={"source": "email"}),
            Document(content="The app crashes on startup."),
        ]
        result = component.run(documents=documents)
        component._agent.predict_batch.assert_called_once_with(
            ["I was charged twice.", "The app crashes on startup."], QUESTIONS, batch_size=4, min_confidence=0.5
        )
        classified = result["documents"]
        assert classified[0].meta == {"source": "email", "laya": _result("billing")["answers"]}
        assert classified[1].meta["laya"]["department"]["choice"] == "technical"
        # Input documents are not mutated
        assert "laya" not in documents[0].meta

    def test_run_classification_field_and_skipped_documents(self, caplog):
        component = LayaDocumentClassifier(questions=QUESTIONS, classification_field="summary")
        component._agent = MagicMock()
        component._agent.predict_batch.return_value = [_result("billing")]
        documents = [
            Document(content="ignored", meta={"summary": "Double charge"}),
            Document(content="no summary"),
        ]
        result = component.run(documents=documents)
        assert component._agent.predict_batch.call_args.args[0] == ["Double charge"]
        assert result["documents"][0].meta["laya"]["department"]["choice"] == "billing"
        assert result["documents"][1] is documents[1]
        assert "has no text to classify" in caplog.text

    def test_run_no_classifiable_documents(self):
        component = LayaDocumentClassifier(questions=QUESTIONS)
        component._agent = MagicMock()
        documents = [Document(content="")]
        result = component.run(documents=documents)
        assert result["documents"] == documents
        component._agent.predict_batch.assert_not_called()

    @patch("haystack_integrations.components.classifiers.laya.document_classifier.load")
    def test_run_warms_up(self, mock_load):
        mock_load.return_value.predict_batch.return_value = [_result("billing")]
        component = LayaDocumentClassifier(questions=QUESTIONS)
        component.run(documents=[Document(content="I was charged twice.")])
        mock_load.assert_called_once()


@pytest.mark.integration
class TestLayaDocumentClassifierInference:
    def test_run(self):
        component = LayaDocumentClassifier(questions=QUESTIONS, device=ComponentDevice.from_str("cpu"))
        result = component.run(
            documents=[
                Document(content="I was charged twice for my subscription, please refund the second payment."),
                Document(content="The app crashes every time I open the settings page."),
            ]
        )
        billing, technical = (document.meta["laya"] for document in result["documents"])
        assert billing["department"]["choice"] == "billing"
        assert technical["department"]["choice"] == "technical"
        assert set(billing["department"]["probabilities"]) == {"billing", "technical"}
        assert billing["refund"]["noul"] > technical["refund"]["noul"]
