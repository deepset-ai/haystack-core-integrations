# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from haystack import ComponentError, DeserializationError, Document, Pipeline
from haystack.utils.device import ComponentDevice, DeviceMap
from thinc.api import NumpyOps, get_current_ops, set_current_ops

from haystack_integrations.components.extractors.spacy import (
    NamedEntityAnnotation,
    SpacyNamedEntityExtractor,
)

MODULE = "haystack_integrations.components.extractors.spacy.named_entity_extractor"
COMPONENT_TYPE = f"{MODULE}.SpacyNamedEntityExtractor"


@pytest.fixture
def raw_texts() -> list:
    return [
        "My name is Clara and I live in Berkeley, California.",
        "I'm Merlin, the happy pig!",
        "New York State declared a state of emergency after the announcement of the end of the world.",
        "",  # Intentionally empty.
    ]


@pytest.fixture
def spacy_annotations() -> list:
    return [
        [
            NamedEntityAnnotation(entity="PERSON", start=11, end=16),
            NamedEntityAnnotation(entity="GPE", start=31, end=39),
            NamedEntityAnnotation(entity="GPE", start=41, end=51),
        ],
        [NamedEntityAnnotation(entity="PERSON", start=4, end=10)],
        [NamedEntityAnnotation(entity="GPE", start=0, end=14)],
        [],
    ]


def test_named_entity_extractor_init():
    extractor = SpacyNamedEntityExtractor(model="en_core_web_trf")
    assert extractor.model_name_or_path == "en_core_web_trf"
    assert extractor.pipeline_kwargs == {}
    assert not extractor.initialized


def test_init_raises_with_multiple_devices():
    device = ComponentDevice.from_multiple(DeviceMap.from_dict({"layer1": "cpu", "layer2": "cuda:0"}))
    with pytest.raises(ValueError, match="only supports inference on single devices"):
        SpacyNamedEntityExtractor(model="en_core_web_trf", device=device)


def test_named_entity_extractor_to_dict():
    extractor = SpacyNamedEntityExtractor(
        model="en_core_web_trf",
        device=ComponentDevice.from_str("cuda:1"),
    )

    serde_data = extractor.to_dict()
    assert serde_data == {
        "type": COMPONENT_TYPE,
        "init_parameters": {
            "model": "en_core_web_trf",
            "device": {"type": "single", "device": "cuda:1"},
            "pipeline_kwargs": {},
        },
    }


def test_named_entity_extractor_to_dict_with_parameters():
    extractor = SpacyNamedEntityExtractor(
        model="en_core_web_trf",
        device=ComponentDevice.from_str("cuda:1"),
        pipeline_kwargs={"n_process": 2},
    )

    serde_data = extractor.to_dict()
    assert serde_data == {
        "type": COMPONENT_TYPE,
        "init_parameters": {
            "model": "en_core_web_trf",
            "device": {"type": "single", "device": "cuda:1"},
            "pipeline_kwargs": {"n_process": 2},
        },
    }


def test_named_entity_extractor_from_dict():
    data = {
        "type": COMPONENT_TYPE,
        "init_parameters": {
            "model": "en_core_web_trf",
            "device": None,
            "pipeline_kwargs": None,
        },
    }
    extractor = SpacyNamedEntityExtractor.from_dict(data)

    assert extractor.model_name_or_path == "en_core_web_trf"
    assert extractor.pipeline_kwargs == {}


def test_named_entity_extractor_from_dict_no_default_parameters():
    data = {
        "type": COMPONENT_TYPE,
        "init_parameters": {"model": "en_core_web_trf"},
    }
    extractor = SpacyNamedEntityExtractor.from_dict(data)

    assert extractor.model_name_or_path == "en_core_web_trf"
    assert extractor.device == ComponentDevice.resolve_device(None)


def test_named_entity_extractor_serde():
    extractor = SpacyNamedEntityExtractor(
        model="en_core_web_trf",
        device=ComponentDevice.from_str("cuda:1"),
    )

    serde_data = extractor.to_dict()
    new_extractor = SpacyNamedEntityExtractor.from_dict(serde_data)

    assert new_extractor.model_name_or_path == extractor.model_name_or_path
    assert new_extractor.device == extractor.device

    with pytest.raises(DeserializationError, match=r"Couldn't deserialize"):
        serde_data["init_parameters"].pop("model")
        _ = SpacyNamedEntityExtractor.from_dict(serde_data)


def test_named_entity_extractor_serde_none_device():
    extractor = SpacyNamedEntityExtractor(model="en_core_web_trf", device=None)

    serde_data = extractor.to_dict()
    new_extractor = SpacyNamedEntityExtractor.from_dict(serde_data)

    assert new_extractor.model_name_or_path == extractor.model_name_or_path
    assert new_extractor.device == extractor.device


# tests for SpacyNamedEntityExtractor serialization/deserialization in a pipeline
def test_named_entity_extractor_pipeline_serde(tmp_path):
    extractor = SpacyNamedEntityExtractor(model="en_core_web_trf")
    p = Pipeline()
    p.add_component(instance=extractor, name="extractor")

    with open(tmp_path / "test_pipeline.yaml", "w") as f:
        p.dump(f)
    with open(tmp_path / "test_pipeline.yaml") as f:
        q = Pipeline.load(f)

    assert p.to_dict() == q.to_dict(), "Pipeline serialization/deserialization with SpacyNamedEntityExtractor failed."


def test_warm_up_skips_when_already_warmed_up():
    extractor = SpacyNamedEntityExtractor(model="en_core_web_trf")
    extractor._warmed_up = True

    with patch(f"{MODULE}.spacy.load") as mock_load:
        extractor.warm_up()
        mock_load.assert_not_called()


def test_warm_up_fails_when_model_load_raises():
    extractor = SpacyNamedEntityExtractor(model="en_core_web_trf")

    with patch(f"{MODULE}.spacy") as mock_spacy:
        mock_spacy.load.side_effect = RuntimeError("boom")

        with pytest.raises(ComponentError, match="failed to initialize"):
            extractor.warm_up()


def test_warm_up_fails_when_model_has_no_ner_component():
    extractor = SpacyNamedEntityExtractor(model="en_core_web_trf")

    with patch(f"{MODULE}.spacy") as mock_spacy:
        mock_spacy.load.return_value.has_pipe.return_value = False

        with pytest.raises(ComponentError, match="failed to initialize"):
            extractor.warm_up()


def test_named_entity_extractor_run():
    """Test the SpacyNamedEntityExtractor.run method with mocked model interaction."""
    documents = [Document(content="My name is Clara and I live in Berkeley, California.")]

    expected_annotations = [
        [
            NamedEntityAnnotation(entity="PERSON", start=11, end=16),
            NamedEntityAnnotation(entity="GPE", start=31, end=39),
            NamedEntityAnnotation(entity="GPE", start=41, end=51),
        ]
    ]

    extractor = SpacyNamedEntityExtractor(model="en_core_web_trf")

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

    extractor = SpacyNamedEntityExtractor(model="en_core_web_trf")

    with patch.object(extractor, "_annotate", return_value=[[]]):
        extractor._warmed_up = True

        with pytest.raises(ComponentError, match="did not return the correct number of annotations"):
            extractor.run(documents=documents)


def test_run_triggers_warm_up_when_not_warmed_up():
    documents = [Document(content="My name is Clara and I live in Berkeley, California.")]
    expected_annotations = [[NamedEntityAnnotation(entity="PERSON", start=11, end=16)]]

    extractor = SpacyNamedEntityExtractor(model="en_core_web_trf")

    with (
        patch.object(extractor, "warm_up") as mock_warm_up,
        patch.object(extractor, "_annotate", return_value=expected_annotations),
    ):
        result = extractor.run(documents=documents)

        mock_warm_up.assert_called_once()
        assert result["documents"][0].meta["named_entities"] == expected_annotations[0]


def test_annotate_raises_when_not_initialized():
    extractor = SpacyNamedEntityExtractor(model="en_core_web_trf")

    with pytest.raises(ComponentError, match="was not initialized"):
        extractor._annotate(["My name is Clara."])


def test_spacy_backend_restores_device_state():
    """
    Verify that SpacyNamedEntityExtractor restores the previous Thinc Ops state after the context manager exits.
    """
    # 1. Setup a custom state
    custom_ops = NumpyOps()
    custom_ops.owner = "test_marker"
    set_current_ops(custom_ops)

    try:
        # 2. Initialize and enter the device-selection context manager
        extractor = SpacyNamedEntityExtractor(model="en_core_web_sm")

        with extractor._select_device():
            # Inside the context, the state might change
            pass

        # 3. Verify state is restored
        final_ops = get_current_ops()
        assert getattr(final_ops, "owner", None) == "test_marker"

    finally:
        # Clean up global state
        set_current_ops(NumpyOps())


@pytest.mark.integration
def test_ner_extractor_init():
    extractor = SpacyNamedEntityExtractor(model="en_core_web_trf")
    extractor.warm_up()
    assert extractor.initialized


@pytest.mark.integration
@pytest.mark.parametrize("batch_size", [1, 3])
def test_ner_extractor(raw_texts, spacy_annotations, batch_size):
    extractor = SpacyNamedEntityExtractor(model="en_core_web_trf")
    extractor.warm_up()

    _extract_and_check_predictions(extractor, raw_texts, spacy_annotations, batch_size)


@pytest.mark.integration
@pytest.mark.parametrize("batch_size", [1, 3])
def test_ner_extractor_in_pipeline(raw_texts, spacy_annotations, batch_size):
    pipeline = Pipeline()
    pipeline.add_component(
        name="ner_extractor",
        instance=SpacyNamedEntityExtractor(model="en_core_web_trf"),
    )

    outputs = pipeline.run(
        {"ner_extractor": {"documents": [Document(content=text) for text in raw_texts], "batch_size": batch_size}}
    )["ner_extractor"]["documents"]
    predicted = [SpacyNamedEntityExtractor.get_stored_annotations(doc) for doc in outputs]
    _check_predictions(predicted, spacy_annotations)


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

    predicted = [SpacyNamedEntityExtractor.get_stored_annotations(doc) for doc in outputs]

    _check_predictions(predicted, expected)


def _check_predictions(predicted, expected):
    assert len(predicted) == len(expected)
    for pred, exp in zip(predicted, expected, strict=True):
        assert len(pred) == len(exp)

        for a, b in zip(pred, exp, strict=True):
            assert a.entity == b.entity
            assert a.start == b.start
            assert a.end == b.end
