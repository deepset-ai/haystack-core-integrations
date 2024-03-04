from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

from optimum.onnxruntime.configuration import AutoOptimizationConfig, OptimizationConfig


class OptimumEmbedderOptimizationMode(Enum):
    """
    [ONXX Optimization modes](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/optimization)
    support by the Optimum Embedders.
    """

    #: Basic general optimizations.
    O1 = "o1"

    #: Basic and extended general optimizations, transformers-specific fusions.
    O2 = "o2"

    #: Same as O2 with Gelu approximation.
    O3 = "o3"

    #: Same as O3 with mixed precision.
    O4 = "o4"

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, string: str) -> "OptimumEmbedderOptimizationMode":
        """
        Create an optimization mode from a string.

        :param string:
            String to convert.
        :returns:
            Optimization mode.
        """
        enum_map = {e.value: e for e in OptimumEmbedderOptimizationMode}
        opt_mode = enum_map.get(string)
        if opt_mode is None:
            msg = f"Unknown optimization mode '{string}'. Supported modes are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return opt_mode


@dataclass(frozen=True)
class OptimumEmbedderOptimizationConfig:
    """
    Configuration for Optimum Embedder Optimization.

    :param mode:
        Optimization mode.
    :param for_gpu:
        Whether to optimize for GPUs.
    """

    mode: OptimumEmbedderOptimizationMode
    for_gpu: bool = True

    def to_optimum_config(self) -> OptimizationConfig:
        """
        Convert the configuration to a Optimum configuration.

        :returns:
            Optimum configuration.
        """
        if self.mode == OptimumEmbedderOptimizationMode.O1:
            return AutoOptimizationConfig.O1(for_gpu=self.for_gpu)
        elif self.mode == OptimumEmbedderOptimizationMode.O2:
            return AutoOptimizationConfig.O2(for_gpu=self.for_gpu)
        elif self.mode == OptimumEmbedderOptimizationMode.O3:
            return AutoOptimizationConfig.O3(for_gpu=self.for_gpu)
        elif self.mode == OptimumEmbedderOptimizationMode.O4:
            return AutoOptimizationConfig.O4(for_gpu=self.for_gpu)
        else:
            msg = f"Unknown optimization mode '{self.mode}'"
            raise ValueError(msg)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return {
            "mode": str(self.mode),
            "for_gpu": self.for_gpu,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimumEmbedderOptimizationConfig":
        """
        Create an optimization configuration from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Optimization configuration.
        """
        return OptimumEmbedderOptimizationConfig(
            mode=OptimumEmbedderOptimizationMode.from_str(data["mode"]),
            for_gpu=data["for_gpu"],
        )
