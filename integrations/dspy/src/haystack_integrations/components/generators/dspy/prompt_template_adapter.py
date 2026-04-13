import typing


class PromptTemplateAdapter:
    """Makes DSPy base prompts better with custom model templates, for an enhanced text output."""

    templates: typing.ClassVar = {
        "mistral": "[INST] {prompt} [/INST]",
        "llama": "<s>[INST] {prompt} [/INST]",
        "chatml": "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant",
        "default": "{prompt}",
    }
    # Templates currently made for only 3 models as samples. This can be extended.
    # (function written below

    def __init__(self, model_family: str = "default"):
        if model_family not in self.templates:
            self.template = self.templates["default"]
        else:
            self.template = self.templates[model_family]

        self.model_family = model_family

    def wrap(self, prompt: str) -> str:
        """Wqrap the prompt according to the model family template."""
        return self.template.format(prompt=prompt)
