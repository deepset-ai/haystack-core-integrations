from unittest.mock import Mock, patch

from vertexai.preview.vision_models import ImageGenerationResponse

from google_vertex_haystack.generators.image_generator import VertexAIImageGenerator


@patch("google_vertex_haystack.generators.image_generator.authenticate")
@patch("google_vertex_haystack.generators.image_generator.ImageGenerationModel")
def test_init(mock_model_class, mock_authenticate):
    generator = VertexAIImageGenerator(
        model="imagetext",
        project_id="myproject-123456",
        api_key="my_api_key",
        guidance_scale=12,
        number_of_images=3,
    )
    mock_authenticate.assert_called_once_with(project_id="myproject-123456", api_key="my_api_key", location=None)
    mock_model_class.from_pretrained.assert_called_once_with("imagetext")
    assert generator._model_name == "imagetext"
    assert generator._project_id == "myproject-123456"
    assert generator._api_key == "my_api_key"
    assert generator._location is None
    assert generator._kwargs == {
        "guidance_scale": 12,
        "number_of_images": 3,
    }


@patch("google_vertex_haystack.generators.image_generator.authenticate")
@patch("google_vertex_haystack.generators.image_generator.ImageGenerationModel")
def test_to_dict(_mock_model_class, _mock_authenticate):
    generator = VertexAIImageGenerator(
        model="imagetext",
        project_id="myproject-123456",
        guidance_scale=12,
        number_of_images=3,
    )
    assert generator.to_dict() == {
        "type": "google_vertex_haystack.generators.image_generator.VertexAIImageGenerator",
        "init_parameters": {
            "model": "imagetext",
            "project_id": "myproject-123456",
            "api_key": "",
            "location": None,
            "guidance_scale": 12,
            "number_of_images": 3,
        },
    }


@patch("google_vertex_haystack.generators.image_generator.authenticate")
@patch("google_vertex_haystack.generators.image_generator.ImageGenerationModel")
def test_from_dict(_mock_model_class, _mock_authenticate):
    generator = VertexAIImageGenerator.from_dict(
        {
            "type": "google_vertex_haystack.generators.image_generator.VertexAIImageGenerator",
            "init_parameters": {
                "model": "imagetext",
                "project_id": "myproject-123456",
                "api_key": "",
                "location": None,
                "guidance_scale": 12,
                "number_of_images": 3,
            },
        }
    )
    assert generator._model_name == "imagetext"
    assert generator._project_id == "myproject-123456"
    assert generator._api_key == ""
    assert generator._location is None
    assert generator._kwargs == {
        "guidance_scale": 12,
        "number_of_images": 3,
    }


@patch("google_vertex_haystack.generators.image_generator.authenticate")
@patch("google_vertex_haystack.generators.image_generator.ImageGenerationModel")
def test_run_calls_generate_images(mock_model_class, _mock_authenticate):
    mock_model = Mock()
    mock_model.generate_images.return_value = ImageGenerationResponse(images=[])
    mock_model_class.from_pretrained.return_value = mock_model
    generator = VertexAIImageGenerator(
        model="imagetext",
        project_id="myproject-123456",
        guidance_scale=12,
        number_of_images=3,
    )

    prompt = "Generate an image of a dog"
    negative_prompt = "Generate an image of a cat"
    generator.run(prompt=prompt, negative_prompt=negative_prompt)

    mock_model.generate_images.assert_called_once_with(
        prompt=prompt, negative_prompt=negative_prompt, guidance_scale=12, number_of_images=3
    )
