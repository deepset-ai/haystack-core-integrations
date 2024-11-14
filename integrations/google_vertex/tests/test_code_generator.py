from unittest.mock import Mock, patch

from vertexai.language_models import TextGenerationResponse

from haystack_integrations.components.generators.google_vertex import VertexAICodeGenerator


@patch("haystack_integrations.components.generators.google_vertex.code_generator.vertexai")
@patch("haystack_integrations.components.generators.google_vertex.code_generator.CodeGenerationModel")
def test_init(mock_model_class, mock_vertexai):
    generator = VertexAICodeGenerator(
        model="code-bison", project_id="myproject-123456", candidate_count=3, temperature=0.5
    )
    mock_vertexai.init.assert_called_once_with(project="myproject-123456", location=None)
    mock_model_class.from_pretrained.assert_called_once_with("code-bison")
    assert generator._model_name == "code-bison"
    assert generator._project_id == "myproject-123456"
    assert generator._location is None
    assert generator._kwargs == {"candidate_count": 3, "temperature": 0.5}


@patch("haystack_integrations.components.generators.google_vertex.code_generator.vertexai")
@patch("haystack_integrations.components.generators.google_vertex.code_generator.CodeGenerationModel")
def test_to_dict(_mock_model_class, _mock_vertexai):
    generator = VertexAICodeGenerator(model="code-bison", candidate_count=3, temperature=0.5)
    assert generator.to_dict() == {
        "type": "haystack_integrations.components.generators.google_vertex.code_generator.VertexAICodeGenerator",
        "init_parameters": {
            "model": "code-bison",
            "project_id": None,
            "location": None,
            "candidate_count": 3,
            "temperature": 0.5,
        },
    }


@patch("haystack_integrations.components.generators.google_vertex.code_generator.vertexai")
@patch("haystack_integrations.components.generators.google_vertex.code_generator.CodeGenerationModel")
def test_from_dict(_mock_model_class, _mock_vertexai):
    generator = VertexAICodeGenerator.from_dict(
        {
            "type": "haystack_integrations.components.generators.google_vertex.code_generator.VertexAICodeGenerator",
            "init_parameters": {
                "model": "code-bison",
                "project_id": None,
                "location": None,
                "candidate_count": 2,
                "temperature": 0.5,
            },
        }
    )
    assert generator._model_name == "code-bison"
    assert generator._project_id is None
    assert generator._location is None
    assert generator._kwargs == {"candidate_count": 2, "temperature": 0.5}
    assert generator._model is not None


@patch("haystack_integrations.components.generators.google_vertex.code_generator.vertexai")
@patch("haystack_integrations.components.generators.google_vertex.code_generator.CodeGenerationModel")
def test_run_calls_predict(mock_model_class, _mock_vertexai):
    mock_model = Mock()
    mock_model.predict.return_value = TextGenerationResponse("answer", None)
    mock_model_class.from_pretrained.return_value = mock_model
    generator = VertexAICodeGenerator(model="code-bison", candidate_count=1, temperature=0.5)

    prefix = "def print_json(data):\n"
    generator.run(prefix=prefix)

    mock_model.predict.assert_called_once()
    assert len(mock_model.predict.call_args.kwargs) == 4
    assert mock_model.predict.call_args.kwargs["prefix"] == prefix
    assert mock_model.predict.call_args.kwargs["suffix"] is None
    assert mock_model.predict.call_args.kwargs["candidate_count"] == 1
    assert mock_model.predict.call_args.kwargs["temperature"] == 0.5
