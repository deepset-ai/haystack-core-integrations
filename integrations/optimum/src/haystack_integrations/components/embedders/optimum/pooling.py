from enum import Enum


class OptimumEmbedderPooling(Enum):
    """
    Pooling Modes support by the Optimum Embedders.
    """

    #: Perform CLS Pooling on the output of the embedding model
    #: using the first token (CLS token).
    CLS = "cls"

    #: Perform Mean Pooling on the output of the embedding model.
    MEAN = "mean"

    #: Perform Max Pooling on the output of the embedding model
    #: using the maximum value in each dimension over all the tokens.
    MAX = "max"

    #: Perform mean-pooling on the output of the embedding model but
    #: divide by the square root of the sequence length.
    MEAN_SQRT_LEN = "mean_sqrt_len"

    #: Perform weighted (position) mean pooling on the output of the
    #: embedding model. See https://arxiv.org/abs/2202.08904.
    WEIGHTED_MEAN = "weighted_mean"

    #: Perform Last Token Pooling on the output of the embedding model.
    #: See https://arxiv.org/abs/2202.08904 & https://arxiv.org/abs/2201.10005.
    LAST_TOKEN = "last_token"

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, string: str) -> "OptimumEmbedderPooling":
        """
        Create a pooling mode from a string.

        :param string:
            The string to convert.
        :returns:
            The pooling mode.
        """
        enum_map = {e.value: e for e in OptimumEmbedderPooling}
        pooling_mode = enum_map.get(string)
        if pooling_mode is None:
            msg = f"Unknown Pooling mode '{string}'. Supported modes are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return pooling_mode
