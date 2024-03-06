from enum import Enum


class NvidiaEmbeddingModel(Enum):
    """
    [NVIDIA AI Foundation models](https://catalog.ngc.nvidia.com/ai-foundation-models)
    used for generating embeddings.
    """

    #: [Retrieval QA Embedding Model](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/nvolve-40k).
    NVOLVE_40K = "playground_nvolveqa_40k"

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, string: str) -> "NvidiaEmbeddingModel":
        """
        Create an embedding model from a string.

        :param string:
            String to convert.
        :returns:
            Embedding model.
        """
        enum_map = {e.value: e for e in NvidiaEmbeddingModel}
        emb_model = enum_map.get(string)
        if emb_model is None:
            msg = f"Unknown embedding model '{string}'. Supported modes are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return emb_model
