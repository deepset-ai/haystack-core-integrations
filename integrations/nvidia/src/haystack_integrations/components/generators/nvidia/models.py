# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from enum import Enum


class NvidiaGeneratorModel(Enum):
    """
    Generator models supported by NvidiaGenerator and NvidiaChatGenerator.
    """

    NV_LLAMA2_RLHF_70B = "playground_nv_llama2_rlhf_70b"
    STEERLM_LLAMA_70B = "playground_steerlm_llama_70b"
    NEMOTRON_STEERLM_8B = "playground_nemotron_steerlm_8b"
    NEMOTRON_QA_8B = "playground_nemotron_qa_8b"

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, string: str) -> "NvidiaGeneratorModel":
        """
        Create a generator model from a string.

        :param string:
            String to convert.
        :returns:
            A generator model.
        """
        enum_map = {e.value: e for e in NvidiaGeneratorModel}
        models = enum_map.get(string)
        if models is None:
            msg = f"Unknown model '{string}'. Supported models are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return models
