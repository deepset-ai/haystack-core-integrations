from unittest.mock import Mock, patch

import pytest
from haystack.dataclasses.byte_stream import ByteStream

from google_vertex_haystack.generators.captioner import VertexAIImageCaptioner


@patch("google_vertex_haystack.generators.captioner.authenticate")
@patch("google_vertex_haystack.generators.captioner.ImageTextModel")
def test_init(mock_model_class, mock_authenticate):
    captioner = VertexAIImageCaptioner(
        model="imagetext", project_id="myproject-123456", api_key="my_api_key", number_of_results=1, language="it"
    )
    mock_authenticate.assert_called_once_with(project_id="myproject-123456", api_key="my_api_key", location=None)
    mock_model_class.from_pretrained.assert_called_once_with("imagetext")
    assert captioner._model_name == "imagetext"
    assert captioner._project_id == "myproject-123456"
    assert captioner._api_key == "my_api_key"
    assert captioner._location is None
    assert captioner._kwargs == {"number_of_results": 1, "language": "it"}


@patch("google_vertex_haystack.generators.captioner.authenticate")
@patch("google_vertex_haystack.generators.captioner.ImageTextModel")
def test_to_dict(_mock_model_class, _mock_authenticate):
    captioner = VertexAIImageCaptioner(
        model="imagetext", project_id="myproject-123456", number_of_results=1, language="it"
    )
    assert captioner.to_dict() == {
        "type": "google_vertex_haystack.generators.captioner.VertexAIImageCaptioner",
        "init_parameters": {
            "model": "imagetext",
            "project_id": "myproject-123456",
            "api_key": "",
            "location": None,
            "number_of_results": 1,
            "language": "it",
        },
    }


@patch("google_vertex_haystack.generators.captioner.authenticate")
@patch("google_vertex_haystack.generators.captioner.ImageTextModel")
def test_from_dict(_mock_model_class, _mock_authenticate):
    captioner = VertexAIImageCaptioner.from_dict(
        {
            "type": "google_vertex_haystack.generators.captioner.VertexAIImageCaptioner",
            "init_parameters": {
                "model": "imagetext",
                "project_id": "myproject-123456",
                "api_key": "",
                "number_of_results": 1,
                "language": "it",
            },
        }
    )
    assert captioner._model_name == "imagetext"
    assert captioner._project_id == "myproject-123456"
    assert captioner._api_key == ""
    assert captioner._location is None
    assert captioner._kwargs == {"number_of_results": 1, "language": "it"}
    assert captioner._model is not None


@patch("google_vertex_haystack.generators.captioner.authenticate")
@patch("google_vertex_haystack.generators.captioner.ImageTextModel")
def test_run_calls_get_captions(mock_model_class, _mock_authenticate):
    mock_model = Mock()
    mock_model_class.from_pretrained.return_value = mock_model
    captioner = VertexAIImageCaptioner(
        model="imagetext", project_id="myproject-123456", number_of_results=1, language="it"
    )

    image = ByteStream(data=b"image data")
    captioner.run(image=image)
    mock_model.get_captions.assert_called_once()
    assert len(mock_model.get_captions.call_args.kwargs) == 3
    assert mock_model.get_captions.call_args.kwargs["image"]._image_bytes == image.data
    assert mock_model.get_captions.call_args.kwargs["number_of_results"] == 1
    assert mock_model.get_captions.call_args.kwargs["language"] == "it"
