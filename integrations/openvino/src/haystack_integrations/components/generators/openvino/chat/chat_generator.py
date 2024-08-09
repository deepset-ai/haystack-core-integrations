# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
# pylint: disable=too-many-public-methods
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
from haystack.dataclasses import StreamingChunk
from haystack.lazy_imports import LazyImport
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoTokenizer

with LazyImport("Install openvino using 'pip install optimum[openvino]'") as ov_import:
    from optimum.intel.openvino import OVModelForCausalLM

DEFAULT_OV_CONFIG = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": "1", "CACHE_DIR": ""}


class OpenVINOChatGenerator(HuggingFaceLocalChatGenerator):
    """
    A Chat Generator component that uses OpenVINO models to generate chat responses locally.

    The `OpenVINOChatGenerator class is a component designed for generating chat responses using models from
    Hugging Face's model hub. It is tailored for local runtime text generation tasks and provides a convenient interface
    for working with chat-based models, such as `HuggingFaceH4/zephyr-7b-beta` or `meta-llama/Llama-2-7b-chat-hf`
    etc.

    Usage example:
    ```python
    from haystack.components.generators.chat import OpenVINOChatGenerator
    from haystack.dataclasses import ChatMessage

    generator = OpenVINOChatGenerator(model="HuggingFaceH4/zephyr-7b-beta")
    generator.warm_up()
    messages = [ChatMessage.from_user("What's Natural Language Processing? Be brief.")]
    print(generator.run(messages))
    ```

    ```
    {'replies':
        [ChatMessage(content=' Natural Language Processing (NLP) is a subfield of artificial intelligence that deals
        with the interaction between computers and human language. It enables computers to understand, interpret, and
        generate human language in a valuable way. NLP involves various techniques such as speech recognition, text
        analysis, sentiment analysis, and machine translation. The ultimate goal is to make it easier for computers to
        process and derive meaning from human language, improving communication between humans and machines.',
        role=<ChatRole.ASSISTANT: 'assistant'>,
        name=None,
        meta={'finish_reason': 'stop', 'index': 0, 'model':
              'mistralai/Mistral-7B-Instruct-v0.2',
              'usage': {'completion_tokens': 90, 'prompt_tokens': 19, 'total_tokens': 109}})
              ]
    }
    ```
    """

    def __init__(
        self,
        model: str = "microsoft/Phi-3-mini-4k-instruct",
        device_openvino: str = "cpu",
        ov_config: dict = DEFAULT_OV_CONFIG,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        huggingface_pipeline_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        Creates an instance of a HuggingFaceLocalGenerator.

        :param model: The name of a Hugging Face model or path of OpenVINO model for text generation.
        :param device: The device on which the model is loaded.
        :param generation_kwargs: A dictionary containing keyword arguments to customize text generation.
            Some examples: `max_length`, `max_new_tokens`, `temperature`, `top_k`, `top_p`,...
            See Hugging Face's documentation for more information:
            - [customize-text-generation](https://huggingface.co/docs/transformers/main/en/generation_strategies#customize-text-generation)
            - [transformers.GenerationConfig](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig)
        :param huggingface_pipeline_kwargs: Dictionary containing keyword arguments used to initialize the
            Hugging Face pipeline for text generation.
            These keyword arguments provide fine-grained control over the Hugging Face pipeline.
            In case of duplication, these kwargs override `model`, `task`, `device`, and `token` init parameters.
            See Hugging Face's [documentation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline.task)
            for more information on the available kwargs.
            In this dictionary, you can also include `model_kwargs` to specify the kwargs for model initialization:
            [transformers.PreTrainedModel.from_pretrained](https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained)
        :param stop_words: A list of stop words. If any one of the stop words is generated, the generation is stopped.
            If you provide this parameter, you should not specify the `stopping_criteria` in `generation_kwargs`.
            For some chat models, the output includes both the new text and the original prompt.
            In these cases, it's important to make sure your prompt has no stop words.
        :param streaming_callback: An optional callable for handling streaming responses.
        """
        ov_import.check()
        super().__init__(
            model=model,
            task="text-generation",
            generation_kwargs=generation_kwargs,
            huggingface_pipeline_kwargs=huggingface_pipeline_kwargs,
            stop_words=stop_words,
            streaming_callback=streaming_callback,
        )

        tokenizer_kwargs = tokenizer_kwargs or {}

        def require_model_export(model_id: str, revision: Any = None, subfolder: Any = None) -> bool:
            model_dir = Path(model_id)
            if subfolder is not None:
                model_dir = model_dir / subfolder
            if model_dir.is_dir():
                return (
                    not (model_dir / "openvino_model.xml").exists() or not (model_dir / "openvino_model.bin").exists()
                )
            hf_api = HfApi()
            try:
                model_info = hf_api.model_info(model_id, revision=revision or "main")
                normalized_subfolder = None if subfolder is None else Path(subfolder).as_posix()
                model_files = [
                    file.rfilename
                    for file in model_info.siblings
                    if normalized_subfolder is None or file.rfilename.startswith(normalized_subfolder)
                ]
                ov_model_path = (
                    "openvino_model.xml" if subfolder is None else f"{normalized_subfolder}/openvino_model.xml"
                )
                return ov_model_path not in model_files or ov_model_path.replace(".xml", ".bin") not in model_files
            except Exception:
                return True

        if require_model_export(model):
            ov_model = OVModelForCausalLM.from_pretrained(
                model,
                device=device_openvino,
                ov_config=ov_config,
                config=AutoConfig.from_pretrained(model, trust_remote_code=True),
                trust_remote_code=True,
                export=True,
            )
        else:
            ov_model = OVModelForCausalLM.from_pretrained(
                model,
                device=device_openvino,
                ov_config=ov_config,
                config=AutoConfig.from_pretrained(model, trust_remote_code=True),
                trust_remote_code=True,
            )

        self.huggingface_pipeline_kwargs["model"] = ov_model
        self.huggingface_pipeline_kwargs["tokenizer"] = AutoTokenizer.from_pretrained(model, **tokenizer_kwargs)
