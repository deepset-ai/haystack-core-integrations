from unittest.mock import MagicMock, Mock, patch

from vertexai.language_models import GroundingSource

from google_vertex_haystack.generators.text_generator import VertexAITextGenerator


@patch("google_vertex_haystack.generators.text_generator.authenticate")
@patch("google_vertex_haystack.generators.text_generator.TextGenerationModel")
def test_init(mock_model_class, mock_authenticate):
    grounding_source = GroundingSource.VertexAISearch("1234", "us-central-1")
    generator = VertexAITextGenerator(
        model="text-bison",
        project_id="myproject-123456",
        api_key="my_api_key",
        temperature=0.2,
        grounding_source=grounding_source,
    )
    mock_authenticate.assert_called_once_with(project_id="myproject-123456", api_key="my_api_key", location=None)
    mock_model_class.from_pretrained.assert_called_once_with("text-bison")
    assert generator._model_name == "text-bison"
    assert generator._project_id == "myproject-123456"
    assert generator._api_key == "my_api_key"
    assert generator._location is None
    assert generator._kwargs == {"temperature": 0.2, "grounding_source": grounding_source}


@patch("google_vertex_haystack.generators.text_generator.authenticate")
@patch("google_vertex_haystack.generators.text_generator.TextGenerationModel")
def test_to_dict(_mock_model_class, _mock_authenticate):
    grounding_source = GroundingSource.VertexAISearch("1234", "us-central-1")
    generator = VertexAITextGenerator(
        model="text-bison", project_id="myproject-123456", temperature=0.2, grounding_source=grounding_source
    )
    assert generator.to_dict() == {
        "type": "google_vertex_haystack.generators.text_generator.VertexAITextGenerator",
        "init_parameters": {
            "model": "text-bison",
            "project_id": "myproject-123456",
            "api_key": "",
            "location": None,
            "temperature": 0.2,
            "grounding_source": {
                "type": "vertexai.language_models._language_models.VertexAISearch",
                "init_parameters": {
                    "location": "us-central-1",
                    "data_store_id": "1234",
                    "project": None,
                    "disable_attribution": False,
                },
            },
        },
    }


@patch("google_vertex_haystack.generators.text_generator.authenticate")
@patch("google_vertex_haystack.generators.text_generator.TextGenerationModel")
def test_from_dict(_mock_model_class, _mock_authenticate):
    generator = VertexAITextGenerator.from_dict(
        {
            "type": "google_vertex_haystack.generators.text_generator.VertexAITextGenerator",
            "init_parameters": {
                "model": "text-bison",
                "project_id": "myproject-123456",
                "api_key": "",
                "location": None,
                "temperature": 0.2,
                "grounding_source": {
                    "type": "vertexai.language_models._language_models.VertexAISearch",
                    "init_parameters": {
                        "location": "us-central-1",
                        "data_store_id": "1234",
                        "project": None,
                        "disable_attribution": False,
                    },
                },
            },
        }
    )
    assert generator._model_name == "text-bison"
    assert generator._project_id == "myproject-123456"
    assert generator._api_key == ""
    assert generator._location is None
    assert generator._kwargs == {
        "temperature": 0.2,
        "grounding_source": GroundingSource.VertexAISearch("1234", "us-central-1"),
    }


@patch("google_vertex_haystack.generators.text_generator.authenticate")
@patch("google_vertex_haystack.generators.text_generator.TextGenerationModel")
def test_run_calls_get_captions(mock_model_class, _mock_authenticate):
    mock_model = Mock()
    mock_model.predict.return_value = MagicMock()
    mock_model_class.from_pretrained.return_value = mock_model
    grounding_source = GroundingSource.VertexAISearch("1234", "us-central-1")
    generator = VertexAITextGenerator(
        model="text-bison", project_id="myproject-123456", temperature=0.2, grounding_source=grounding_source
    )

    prompt = "What is the answer?"
    generator.run(prompt=prompt)

    mock_model.predict.assert_called_once_with(prompt=prompt, temperature=0.2, grounding_source=grounding_source)
