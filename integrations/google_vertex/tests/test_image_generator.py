from unittest.mock import Mock, patch

from vertexai.vision_models import ImageGenerationResponse

from haystack_integrations.components.generators.google_vertex import VertexAIImageGenerator


@patch("haystack_integrations.components.generators.google_vertex.image_generator.vertexai")
@patch("haystack_integrations.components.generators.google_vertex.image_generator.ImageGenerationModel")
def test_init(mock_model_class, mock_vertexai):
    generator = VertexAIImageGenerator(
        model="imagetext",
        project_id="myproject-123456",
        guidance_scale=12,
        number_of_images=3,
    )
    mock_vertexai.init.assert_called_once_with(project="myproject-123456", location=None)
    mock_model_class.from_pretrained.assert_called_once_with("imagetext")
    assert generator._model_name == "imagetext"
    assert generator._project_id == "myproject-123456"
    assert generator._location is None
    assert generator._kwargs == {
        "guidance_scale": 12,
        "number_of_images": 3,
    }


@patch("haystack_integrations.components.generators.google_vertex.image_generator.vertexai")
@patch("haystack_integrations.components.generators.google_vertex.image_generator.ImageGenerationModel")
def test_to_dict(_mock_model_class, _mock_vertexai):
    generator = VertexAIImageGenerator(
        model="imagetext",
        guidance_scale=12,
        number_of_images=3,
    )
    assert generator.to_dict() == {
        "type": "haystack_integrations.components.generators.google_vertex.image_generator.VertexAIImageGenerator",
        "init_parameters": {
            "model": "imagetext",
            "project_id": None,
            "location": None,
            "guidance_scale": 12,
            "number_of_images": 3,
        },
    }


@patch("haystack_integrations.components.generators.google_vertex.image_generator.vertexai")
@patch("haystack_integrations.components.generators.google_vertex.image_generator.ImageGenerationModel")
def test_from_dict(_mock_model_class, _mock_vertexai):
    generator = VertexAIImageGenerator.from_dict(
        {
            "type": "haystack_integrations.components.generators.google_vertex.image_generator.VertexAIImageGenerator",
            "init_parameters": {
                "model": "imagetext",
                "project_id": None,
                "location": None,
                "guidance_scale": 12,
                "number_of_images": 3,
            },
        }
    )
    assert generator._model_name == "imagetext"
    assert generator._project_id is None
    assert generator._location is None
    assert generator._kwargs == {
        "guidance_scale": 12,
        "number_of_images": 3,
    }


@patch("haystack_integrations.components.generators.google_vertex.image_generator.vertexai")
@patch("haystack_integrations.components.generators.google_vertex.image_generator.ImageGenerationModel")
def test_run_calls_generate_images(mock_model_class, _mock_vertexai):
    mock_model = Mock()
    mock_model.generate_images.return_value = ImageGenerationResponse(images=[])
    mock_model_class.from_pretrained.return_value = mock_model
    generator = VertexAIImageGenerator(
        model="imagetext",
        guidance_scale=12,
        number_of_images=3,
    )

    prompt = "Generate an image of a dog"
    negative_prompt = "Generate an image of a cat"
    generator.run(prompt=prompt, negative_prompt=negative_prompt)

    mock_model.generate_images.assert_called_once_with(
        prompt=prompt, negative_prompt=negative_prompt, guidance_scale=12, number_of_images=3
    )
