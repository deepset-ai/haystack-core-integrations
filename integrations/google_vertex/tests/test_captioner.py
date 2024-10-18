from unittest.mock import Mock, patch

from haystack.dataclasses.byte_stream import ByteStream

from haystack_integrations.components.generators.google_vertex import VertexAIImageCaptioner


@patch("haystack_integrations.components.generators.google_vertex.captioner.vertexai")
@patch("haystack_integrations.components.generators.google_vertex.captioner.ImageTextModel")
def test_init(mock_model_class, mock_vertexai):
    captioner = VertexAIImageCaptioner(
        model="imagetext", project_id="myproject-123456", number_of_results=1, language="it"
    )
    mock_vertexai.init.assert_called_once_with(project="myproject-123456", location=None)
    mock_model_class.from_pretrained.assert_called_once_with("imagetext")
    assert captioner._model_name == "imagetext"
    assert captioner._project_id == "myproject-123456"
    assert captioner._location is None
    assert captioner._kwargs == {"number_of_results": 1, "language": "it"}


@patch("haystack_integrations.components.generators.google_vertex.captioner.vertexai")
@patch("haystack_integrations.components.generators.google_vertex.captioner.ImageTextModel")
def test_to_dict(_mock_model_class, _mock_vertexai):
    captioner = VertexAIImageCaptioner(model="imagetext", number_of_results=1, language="it")
    assert captioner.to_dict() == {
        "type": "haystack_integrations.components.generators.google_vertex.captioner.VertexAIImageCaptioner",
        "init_parameters": {
            "model": "imagetext",
            "project_id": None,
            "location": None,
            "number_of_results": 1,
            "language": "it",
        },
    }


@patch("haystack_integrations.components.generators.google_vertex.captioner.vertexai")
@patch("haystack_integrations.components.generators.google_vertex.captioner.ImageTextModel")
def test_from_dict(_mock_model_class, _mock_vertexai):
    captioner = VertexAIImageCaptioner.from_dict(
        {
            "type": "haystack_integrations.components.generators.google_vertex.captioner.VertexAIImageCaptioner",
            "init_parameters": {
                "model": "imagetext",
                "project_id": None,
                "location": None,
                "number_of_results": 1,
                "language": "it",
            },
        }
    )
    assert captioner._model_name == "imagetext"
    assert captioner._project_id is None
    assert captioner._location is None
    assert captioner._kwargs == {"number_of_results": 1, "language": "it"}
    assert captioner._model is not None


@patch("haystack_integrations.components.generators.google_vertex.captioner.vertexai")
@patch("haystack_integrations.components.generators.google_vertex.captioner.ImageTextModel")
def test_run_calls_get_captions(mock_model_class, _mock_vertexai):
    mock_model = Mock()
    mock_model_class.from_pretrained.return_value = mock_model
    captioner = VertexAIImageCaptioner(model="imagetext", number_of_results=1, language="it")

    image = ByteStream(data=b"image data")
    captioner.run(image=image)
    mock_model.get_captions.assert_called_once()
    assert len(mock_model.get_captions.call_args.kwargs) == 3
    assert mock_model.get_captions.call_args.kwargs["image"]._image_bytes == image.data
    assert mock_model.get_captions.call_args.kwargs["number_of_results"] == 1
    assert mock_model.get_captions.call_args.kwargs["language"] == "it"
