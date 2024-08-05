import logging
from typing import Any, Dict, List, Optional

from gradientai import Gradient
from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)


@component
class GradientGenerator:
    """
    LLM Generator interfacing [Gradient AI](https://gradient.ai/).

    Queries the LLM using Gradient AI's SDK ('gradientai' package).
    See [Gradient AI API](https://docs.gradient.ai/docs/sdk-quickstart) for more details.

    Usage example:
    ```python
    from haystack_integrations.components.generators.gradient import GradientGenerator

    llm = GradientGenerator(base_model_slug="llama2-7b-chat")
    llm.warm_up()
    print(llm.run(prompt="What is the meaning of life?"))
    # Output: {'replies': ['42']}
    ```
    """

    def __init__(
        self,
        *,
        access_token: Secret = Secret.from_env_var(
            "GRADIENT_ACCESS_TOKEN"
        ),  # noqa: B008
        base_model_slug: Optional[str] = None,
        host: Optional[str] = None,
        max_generated_token_count: Optional[int] = None,
        model_adapter_id: Optional[str] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        workspace_id: Secret = Secret.from_env_var(
            "GRADIENT_WORKSPACE_ID"
        ),  # noqa: B008
    ) -> None:
        """
        Create a GradientGenerator component.

        :param access_token: The Gradient access token as a `Secret`. If not provided it's read from the environment
                             variable `GRADIENT_ACCESS_TOKEN`.
        :param base_model_slug: The base model slug to use.
        :param host: The Gradient host. By default, it uses [Gradient AI](https://api.gradient.ai/).
        :param max_generated_token_count: The maximum number of tokens to generate.
        :param model_adapter_id: The model adapter ID to use.
        :param temperature: The temperature to use.
        :param top_k: The top k to use.
        :param top_p: The top p to use.
        :param workspace_id: The Gradient workspace ID as a `Secret`. If not provided it's read from the environment
                             variable `GRADIENT_WORKSPACE_ID`.
        """
        self._access_token = access_token
        self._base_model_slug = base_model_slug
        self._host = host
        self._max_generated_token_count = max_generated_token_count
        self._model_adapter_id = model_adapter_id
        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p
        self._workspace_id = workspace_id

        has_base_model_slug = base_model_slug is not None and base_model_slug != ""
        has_model_adapter_id = model_adapter_id is not None and model_adapter_id != ""

        if not has_base_model_slug and not has_model_adapter_id:
            msg = "Either base_model_slug or model_adapter_id must be provided."
            raise ValueError(msg)
        if has_base_model_slug and has_model_adapter_id:
            msg = "Only one of base_model_slug or model_adapter_id must be provided."
            raise ValueError(msg)

        if has_base_model_slug:
            self._base_model_slug = base_model_slug
        if has_model_adapter_id:
            self._model_adapter_id = model_adapter_id

        self._gradient = Gradient(
            access_token=access_token.resolve_value(),
            host=host,
            workspace_id=workspace_id.resolve_value(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        return default_to_dict(
            self,
            access_token=self._access_token.to_dict(),
            base_model_slug=self._base_model_slug,
            host=self._host,
            max_generated_token_count=self._max_generated_token_count,
            model_adapter_id=self._model_adapter_id,
            temperature=self._temperature,
            top_k=self._top_k,
            top_p=self._top_p,
            workspace_id=self._workspace_id.to_dict(),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GradientGenerator":
        """
        Deserialize this component from a dictionary.

        :param data: The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """

        deserialize_secrets_inplace(
            data["init_parameters"], keys=["access_token", "workspace_id"]
        )
        return default_from_dict(cls, data)

    def warm_up(self):
        """
        Initializes the LLM model instance if it doesn't exist.
        """
        if not hasattr(self, "_model"):
            if isinstance(self._base_model_slug, str):
                self._model = self._gradient.get_base_model(
                    base_model_slug=self._base_model_slug
                )
            if isinstance(self._model_adapter_id, str):
                self._model = self._gradient.get_model_adapter(
                    model_adapter_id=self._model_adapter_id
                )

    @component.output_types(replies=List[str])
    def run(self, prompt: str):
        """
        Queries the LLM with the prompt to produce replies.

        :param prompt: The prompt to be sent to the generative model.
        """
        resp = self._model.complete(
            query=prompt,
            max_generated_token_count=self._max_generated_token_count,
            temperature=self._temperature,
            top_k=self._top_k,
            top_p=self._top_p,
        )
        return {"replies": [resp.generated_output]}
