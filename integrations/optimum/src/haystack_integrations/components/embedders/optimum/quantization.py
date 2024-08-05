from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

from optimum.onnxruntime.configuration import AutoQuantizationConfig, QuantizationConfig


class OptimumEmbedderQuantizationMode(Enum):
    """
    [Dynamic Quantization modes](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/quantization)
    support by the Optimum Embedders.
    """

    #: Quantization for the ARM64 architecture.
    ARM64 = "arm64"

    #: Quantization with AVX-2 instructions.
    AVX2 = "avx2"

    #: Quantization with AVX-512 instructions.
    AVX512 = "avx512"

    #: Quantization with AVX-512 and VNNI instructions.
    AVX512_VNNI = "avx512_vnni"

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, string: str) -> "OptimumEmbedderQuantizationMode":
        """
        Create an quantization mode from a string.

        :param string:
            String to convert.
        :returns:
            Quantization mode.
        """
        enum_map = {e.value: e for e in OptimumEmbedderQuantizationMode}
        q_mode = enum_map.get(string)
        if q_mode is None:
            msg = f"Unknown quantization mode '{string}'. Supported modes are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return q_mode


@dataclass(frozen=True)
class OptimumEmbedderQuantizationConfig:
    """
    Configuration for Optimum Embedder Quantization.

    :param mode:
        Quantization mode.
    :param per_channel:
        Whether to apply per-channel quantization.
    """

    mode: OptimumEmbedderQuantizationMode
    per_channel: bool = False

    def to_optimum_config(self) -> QuantizationConfig:
        """
        Convert the configuration to a Optimum configuration.

        :returns:
            Optimum configuration.
        """
        if self.mode == OptimumEmbedderQuantizationMode.ARM64:
            return AutoQuantizationConfig.arm64(is_static=False, per_channel=self.per_channel)
        elif self.mode == OptimumEmbedderQuantizationMode.AVX2:
            return AutoQuantizationConfig.avx2(is_static=False, per_channel=self.per_channel)
        elif self.mode == OptimumEmbedderQuantizationMode.AVX512:
            return AutoQuantizationConfig.avx512(is_static=False, per_channel=self.per_channel)
        elif self.mode == OptimumEmbedderQuantizationMode.AVX512_VNNI:
            return AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=self.per_channel)
        else:
            msg = f"Unknown quantization mode '{self.mode}'"
            raise ValueError(msg)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return {
            "mode": str(self.mode),
            "per_channel": self.per_channel,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimumEmbedderQuantizationConfig":
        """
        Create a configuration from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Quantization configuration.
        """
        return OptimumEmbedderQuantizationConfig(
            mode=OptimumEmbedderQuantizationMode.from_str(data["mode"]),
            per_channel=data["per_channel"],
        )
