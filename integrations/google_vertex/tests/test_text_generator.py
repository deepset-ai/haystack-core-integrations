from unittest.mock import MagicMock, Mock, patch

from vertexai.language_models import GroundingSource

from haystack_integrations.components.generators.google_vertex import VertexAITextGenerator


@patch("haystack_integrations.components.generators.google_vertex.text_generator.vertexai")
@patch("haystack_integrations.components.generators.google_vertex.text_generator.TextGenerationModel")
def test_init(mock_model_class, mock_vertexai):
    grounding_source = GroundingSource.VertexAISearch("1234", "us-central-1")
    generator = VertexAITextGenerator(
        model="text-bison", project_id="myproject-123456", temperature=0.2, grounding_source=grounding_source
    )
    mock_vertexai.init.assert_called_once_with(project="myproject-123456", location=None)
    mock_model_class.from_pretrained.assert_called_once_with("text-bison")
    assert generator._model_name == "text-bison"
    assert generator._project_id == "myproject-123456"
    assert generator._location is None
    assert generator._kwargs == {"temperature": 0.2, "grounding_source": grounding_source}


@patch("haystack_integrations.components.generators.google_vertex.text_generator.vertexai")
@patch("haystack_integrations.components.generators.google_vertex.text_generator.TextGenerationModel")
def test_to_dict(_mock_model_class, _mock_vertexai):
    grounding_source = GroundingSource.VertexAISearch("1234", "us-central-1")
    generator = VertexAITextGenerator(model="text-bison", temperature=0.2, grounding_source=grounding_source)
    assert generator.to_dict() == {
        "type": "haystack_integrations.components.generators.google_vertex.text_generator.VertexAITextGenerator",
        "init_parameters": {
            "model": "text-bison",
            "project_id": None,
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


@patch("haystack_integrations.components.generators.google_vertex.text_generator.vertexai")
@patch("haystack_integrations.components.generators.google_vertex.text_generator.TextGenerationModel")
def test_from_dict(_mock_model_class, _mock_vertexai):
    generator = VertexAITextGenerator.from_dict(
        {
            "type": "haystack_integrations.components.generators.google_vertex.text_generator.VertexAITextGenerator",
            "init_parameters": {
                "model": "text-bison",
                "project_id": None,
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
    assert generator._project_id is None
    assert generator._location is None
    assert generator._kwargs == {
        "temperature": 0.2,
        "grounding_source": GroundingSource.VertexAISearch("1234", "us-central-1"),
    }


@patch("haystack_integrations.components.generators.google_vertex.text_generator.vertexai")
@patch("haystack_integrations.components.generators.google_vertex.text_generator.TextGenerationModel")
def test_run_calls_get_captions(mock_model_class, _mock_vertexai):
    mock_model = Mock()
    mock_model.predict.return_value = MagicMock()
    mock_model_class.from_pretrained.return_value = mock_model
    grounding_source = GroundingSource.VertexAISearch("1234", "us-central-1")
    generator = VertexAITextGenerator(model="text-bison", temperature=0.2, grounding_source=grounding_source)

    prompt = "What is the answer?"
    generator.run(prompt=prompt)

    mock_model.predict.assert_called_once_with(prompt=prompt, temperature=0.2, grounding_source=grounding_source)
