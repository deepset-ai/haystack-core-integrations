# SPDX-FileCopyrightText: 2024-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from optimum.onnxruntime.configuration import OptimizationConfig, QuantizationConfig

from haystack_integrations.components.embedders.optimum.optimization import (
    OptimumEmbedderOptimizationConfig,
    OptimumEmbedderOptimizationMode,
)
from haystack_integrations.components.embedders.optimum.quantization import (
    OptimumEmbedderQuantizationConfig,
    OptimumEmbedderQuantizationMode,
)


class TestOptimumEmbedderQuantizationConfig:
    @pytest.mark.parametrize(
        "mode",
        [
            OptimumEmbedderQuantizationMode.ARM64,
            OptimumEmbedderQuantizationMode.AVX2,
            OptimumEmbedderQuantizationMode.AVX512,
            OptimumEmbedderQuantizationMode.AVX512_VNNI,
        ],
    )
    def test_to_optimum_config_returns_quantization_config_for_each_mode(self, mode):
        config = OptimumEmbedderQuantizationConfig(mode=mode, per_channel=True)
        optimum_config = config.to_optimum_config()

        assert isinstance(optimum_config, QuantizationConfig)
        assert optimum_config.is_static is False
        assert optimum_config.per_channel is True


class TestOptimumEmbedderOptimizationConfig:
    @pytest.mark.parametrize(
        "mode",
        [
            OptimumEmbedderOptimizationMode.O1,
            OptimumEmbedderOptimizationMode.O2,
            OptimumEmbedderOptimizationMode.O3,
            OptimumEmbedderOptimizationMode.O4,
        ],
    )
    def test_to_optimum_config_returns_optimization_config_for_each_mode(self, mode):
        config = OptimumEmbedderOptimizationConfig(mode=mode, for_gpu=False)
        optimum_config = config.to_optimum_config()

        assert isinstance(optimum_config, OptimizationConfig)
