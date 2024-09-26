# This example requires GPU support to execute.

from haystack import Pipeline

from haystack_integrations.components.embedders.optimum import (
    OptimumEmbedderOptimizationConfig,
    OptimumEmbedderOptimizationMode,
    OptimumEmbedderPooling,
    OptimumTextEmbedder,
)

pipeline = Pipeline()
embedder = OptimumTextEmbedder(
    model="intfloat/e5-base-v2",
    normalize_embeddings=True,
    onnx_execution_provider="CUDAExecutionProvider",
    optimizer_settings=OptimumEmbedderOptimizationConfig(
        mode=OptimumEmbedderOptimizationMode.O4,
        for_gpu=True,
    ),
    working_dir="/tmp/optimum",
    pooling_mode=OptimumEmbedderPooling.MEAN,
)
pipeline.add_component("embedder", embedder)

results = pipeline.run(
    {
        "embedder": {
            "text": "Ex profunditate antiquae doctrinae, Ad caelos supra semper, Hoc incantamentum evoco, draco apparet, Incantamentum iam transactum est"
        },
    }
)

print(results["embedder"]["embedding"])
