# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import patch

import pytest
from haystack import ComponentError, DeserializationError, Document, Pipeline
from haystack.utils.auth import Secret
from haystack.utils.device import ComponentDevice

from haystack_integrations.components.extractors.transformers import (
    NamedEntityAnnotation,
    TransformersNamedEntityExtractor,
)

COMPONENT_TYPE = (
    "haystack_integrations.components.extractors.transformers.named_entity_extractor.TransformersNamedEntityExtractor"
)


@pytest.fixture
def raw_texts() -> list:
    return [
        "My name is Clara and I live in Berkeley, California.",
        "I'm Merlin, the happy pig!",
        "New York State declared a state of emergency after the announcement of the end of the world.",
        "",  # Intentionally empty.
    ]


@pytest.fixture
def hf_annotations() -> list:
    return [
        [
            NamedEntityAnnotation(entity="PER", start=11, end=16),
            NamedEntityAnnotation(entity="LOC", start=31, end=39),
            NamedEntityAnnotation(entity="LOC", start=41, end=51),
        ],
        [NamedEntityAnnotation(entity="PER", start=4, end=10)],
        [NamedEntityAnnotation(entity="LOC", start=0, end=14)],
        [],
    ]


def test_named_entity_extractor_init():
    _ = TransformersNamedEntityExtractor(model="dslim/bert-base-NER")

    # private model
    _ = TransformersNamedEntityExtractor(model="deepset/bert-base-NER")


def test_named_entity_extractor_to_dict():
    extractor = TransformersNamedEntityExtractor(
        model="dslim/bert-base-NER",
        device=ComponentDevice.from_str("cuda:1"),
    )

    serde_data = extractor.to_dict()
    assert serde_data == {
        "type": COMPONENT_TYPE,
        "init_parameters": {
            "model": "dslim/bert-base-NER",
            "device": {"type": "single", "device": "cuda:1"},
            "pipeline_kwargs": {"model": "dslim/bert-base-NER", "device": "cuda:1", "task": "ner"},
            "token": {"type": "env_var", "env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False},
        },
    }


def test_named_entity_extractor_from_dict():
    data = {
        "type": COMPONENT_TYPE,
        "init_parameters": {
            "model": "dslim/bert-base-NER",
            "device": None,
            "pipeline_kwargs": None,
            "token": {"type": "env_var", "env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False},
        },
    }
    extractor = TransformersNamedEntityExtractor.from_dict(data)

    assert extractor.model_name_or_path == "dslim/bert-base-NER"


def test_named_entity_extractor_serde():
    extractor = TransformersNamedEntityExtractor(
        model="dslim/bert-base-NER",
        device=ComponentDevice.from_str("cuda:1"),
    )

    serde_data = extractor.to_dict()
    new_extractor = TransformersNamedEntityExtractor.from_dict(serde_data)

    assert new_extractor.model_name_or_path == extractor.model_name_or_path
    assert new_extractor.device == extractor.device

    with pytest.raises(DeserializationError, match=r"Couldn't deserialize"):
        serde_data["init_parameters"].pop("model")
        _ = TransformersNamedEntityExtractor.from_dict(serde_data)


def test_to_dict_default(del_hf_env_vars_if_empty):
    component = TransformersNamedEntityExtractor(
        model="dslim/bert-base-NER",
        device=ComponentDevice.from_str("mps"),
    )
    data = component.to_dict()

    assert data == {
        "type": COMPONENT_TYPE,
        "init_parameters": {
            "model": "dslim/bert-base-NER",
            "device": {"type": "single", "device": "mps"},
            "pipeline_kwargs": {"model": "dslim/bert-base-NER", "device": "mps", "task": "ner"},
            "token": {"type": "env_var", "env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False},
        },
    }


def test_to_dict_with_parameters():
    component = TransformersNamedEntityExtractor(
        model="dslim/bert-base-NER",
        device=ComponentDevice.from_str("mps"),
        pipeline_kwargs={"model_kwargs": {"load_in_4bit": True}},
        token=Secret.from_env_var("ENV_VAR", strict=False),
    )
    data = component.to_dict()

    assert data == {
        "type": COMPONENT_TYPE,
        "init_parameters": {
            "model": "dslim/bert-base-NER",
            "device": {"type": "single", "device": "mps"},
            "pipeline_kwargs": {
                "model": "dslim/bert-base-NER",
                "device": "mps",
                "task": "ner",
                "model_kwargs": {"load_in_4bit": True},
            },
            "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
        },
    }


def test_named_entity_extractor_from_dict_no_default_parameters(del_hf_env_vars_if_empty):
    data = {
        "type": COMPONENT_TYPE,
        "init_parameters": {"model": "dslim/bert-base-NER"},
    }
    extractor = TransformersNamedEntityExtractor.from_dict(data)

    assert extractor.model_name_or_path == "dslim/bert-base-NER"
    assert extractor.device == ComponentDevice.resolve_device(None)


# tests for TransformersNamedEntityExtractor serialization/deserialization in a pipeline
def test_named_entity_extractor_pipeline_serde(tmp_path):
    extractor = TransformersNamedEntityExtractor(model="dslim/bert-base-NER")
    p = Pipeline()
    p.add_component(instance=extractor, name="extractor")

    with open(tmp_path / "test_pipeline.yaml", "w") as f:
        p.dump(f)
    with open(tmp_path / "test_pipeline.yaml") as f:
        q = Pipeline.load(f)

    assert p.to_dict() == q.to_dict(), (
        "Pipeline serialization/deserialization with TransformersNamedEntityExtractor failed."
    )


def test_named_entity_extractor_serde_none_device():
    extractor = TransformersNamedEntityExtractor(model="dslim/bert-base-NER", device=None)

    serde_data = extractor.to_dict()
    new_extractor = TransformersNamedEntityExtractor.from_dict(serde_data)

    assert new_extractor.model_name_or_path == extractor.model_name_or_path
    assert new_extractor.device == extractor.device


def test_named_entity_extractor_run():
    """Test the TransformersNamedEntityExtractor.run method with mocked model interaction."""
    documents = [Document(content="My name is Clara and I live in Berkeley, California.")]

    expected_annotations = [
        [
            NamedEntityAnnotation(entity="PER", start=11, end=16, score=0.95),
            NamedEntityAnnotation(entity="LOC", start=31, end=39, score=0.88),
            NamedEntityAnnotation(entity="LOC", start=41, end=51, score=0.92),
        ]
    ]

    extractor = TransformersNamedEntityExtractor(model="dslim/bert-base-NER")

    with patch.object(extractor, "_annotate", return_value=expected_annotations) as mock_annotate:
        extractor.pipeline = "mocked_pipeline"
        extractor._warmed_up = True

        result = extractor.run(documents=documents, batch_size=2)

        mock_annotate.assert_called_once_with(["My name is Clara and I live in Berkeley, California."], batch_size=2)

        assert "documents" in result
        assert len(result["documents"]) == 1

        assert isinstance(result["documents"][0], Document)
        assert result["documents"][0].content == documents[0].content
        assert "named_entities" in result["documents"][0].meta
        assert result["documents"][0].meta["named_entities"] == expected_annotations[0]
        assert "named_entities" not in documents[0].meta


def test_named_entity_extractor_run_fails_with_wrong_number_of_annotations():
    documents = [Document(content="My name is Clara."), Document(content="I'm Merlin, the happy pig!")]

    extractor = TransformersNamedEntityExtractor(model="dslim/bert-base-NER")

    with patch.object(extractor, "_annotate", return_value=[[]]):
        extractor._warmed_up = True

        with pytest.raises(ComponentError, match="did not return the correct number of annotations"):
            extractor.run(documents=documents)


@pytest.mark.integration
def test_ner_extractor_init(del_hf_env_vars_if_empty):
    extractor = TransformersNamedEntityExtractor(model="dslim/bert-base-NER")
    extractor.warm_up()
    assert extractor.initialized


@pytest.mark.integration
@pytest.mark.parametrize("batch_size", [1, 3])
def test_ner_extractor(raw_texts, hf_annotations, batch_size, del_hf_env_vars_if_empty):
    extractor = TransformersNamedEntityExtractor(model="dslim/bert-base-NER")
    extractor.warm_up()

    _extract_and_check_predictions(extractor, raw_texts, hf_annotations, batch_size)


@pytest.mark.integration
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.skipif(
    not os.environ.get("HF_API_TOKEN", None) and not os.environ.get("HF_TOKEN", None),
    reason="Export an env var called HF_API_TOKEN or HF_TOKEN containing the Hugging Face token to run this test.",
)
def test_ner_extractor_private_models(raw_texts, hf_annotations, batch_size):
    extractor = TransformersNamedEntityExtractor(model="deepset/bert-base-NER")
    extractor.warm_up()

    _extract_and_check_predictions(extractor, raw_texts, hf_annotations, batch_size)


@pytest.mark.integration
@pytest.mark.parametrize("batch_size", [1, 3])
def test_ner_extractor_in_pipeline(raw_texts, hf_annotations, batch_size, del_hf_env_vars_if_empty):
    pipeline = Pipeline()
    pipeline.add_component(
        name="ner_extractor",
        instance=TransformersNamedEntityExtractor(model="dslim/bert-base-NER"),
    )

    outputs = pipeline.run(
        {"ner_extractor": {"documents": [Document(content=text) for text in raw_texts], "batch_size": batch_size}}
    )["ner_extractor"]["documents"]
    predicted = [TransformersNamedEntityExtractor.get_stored_annotations(doc) for doc in outputs]
    _check_predictions(predicted, hf_annotations)


def _extract_and_check_predictions(extractor, texts, expected, batch_size):
    docs = [Document(content=text) for text in texts]
    outputs = extractor.run(documents=docs, batch_size=batch_size)["documents"]
    for original_doc, output_doc in zip(docs, outputs, strict=True):
        # we don't modify documents in place
        assert original_doc is not output_doc

        # apart from meta, the documents should be identical
        output_doc_dict = output_doc.to_dict(flatten=False)
        output_doc_dict.pop("meta", None)
        assert original_doc.to_dict() == output_doc_dict

    predicted = [TransformersNamedEntityExtractor.get_stored_annotations(doc) for doc in outputs]

    _check_predictions(predicted, expected)


def _check_predictions(predicted, expected):
    assert len(predicted) == len(expected)
    for pred, exp in zip(predicted, expected, strict=True):
        assert len(pred) == len(exp)

        for a, b in zip(pred, exp, strict=True):
            assert a.entity == b.entity
            assert a.start == b.start
            assert a.end == b.end
