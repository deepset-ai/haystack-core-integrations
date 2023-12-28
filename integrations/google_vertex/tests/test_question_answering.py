from unittest.mock import Mock, patch

from haystack.dataclasses.byte_stream import ByteStream

from google_vertex_haystack.generators.question_answering import VertexAIImageQA


@patch("google_vertex_haystack.generators.question_answering.vertexai")
@patch("google_vertex_haystack.generators.question_answering.ImageTextModel")
def test_init(mock_model_class, mock_vertexai):
    generator = VertexAIImageQA(
        model="imagetext",
        project_id="myproject-123456",
        number_of_results=3,
    )
    mock_vertexai.init.assert_called_once_with(project="myproject-123456", location=None)
    mock_model_class.from_pretrained.assert_called_once_with("imagetext")
    assert generator._model_name == "imagetext"
    assert generator._project_id == "myproject-123456"
    assert generator._location is None
    assert generator._kwargs == {"number_of_results": 3}


@patch("google_vertex_haystack.generators.question_answering.vertexai")
@patch("google_vertex_haystack.generators.question_answering.ImageTextModel")
def test_to_dict(_mock_model_class, _mock_vertexai):
    generator = VertexAIImageQA(
        model="imagetext",
        project_id="myproject-123456",
        number_of_results=3,
    )
    assert generator.to_dict() == {
        "type": "google_vertex_haystack.generators.question_answering.VertexAIImageQA",
        "init_parameters": {
            "model": "imagetext",
            "project_id": "myproject-123456",
            "location": None,
            "number_of_results": 3,
        },
    }


@patch("google_vertex_haystack.generators.question_answering.vertexai")
@patch("google_vertex_haystack.generators.question_answering.ImageTextModel")
def test_from_dict(_mock_model_class, _mock_vertexai):
    generator = VertexAIImageQA.from_dict(
        {
            "type": "google_vertex_haystack.generators.question_answering.VertexAIImageQA",
            "init_parameters": {
                "model": "imagetext",
                "project_id": "myproject-123456",
                "location": None,
                "number_of_results": 3,
            },
        }
    )
    assert generator._model_name == "imagetext"
    assert generator._project_id == "myproject-123456"
    assert generator._location is None
    assert generator._kwargs == {"number_of_results": 3}


@patch("google_vertex_haystack.generators.question_answering.vertexai")
@patch("google_vertex_haystack.generators.question_answering.ImageTextModel")
def test_run_calls_ask_question(mock_model_class, _mock_vertexai):
    mock_model = Mock()
    mock_model.ask_question.return_value = []
    mock_model_class.from_pretrained.return_value = mock_model
    generator = VertexAIImageQA(
        model="imagetext",
        project_id="myproject-123456",
        number_of_results=3,
    )

    image = ByteStream(data=b"image data")
    question = "What is this?"
    generator.run(image=image, question=question)

    mock_model.ask_question.assert_called_once()
    assert len(mock_model.ask_question.call_args.kwargs) == 3
    assert mock_model.ask_question.call_args.kwargs["image"]._image_bytes == image.data
    assert mock_model.ask_question.call_args.kwargs["number_of_results"] == 3
    assert mock_model.ask_question.call_args.kwargs["question"] == question
