from enum import Enum


class EmbeddingTruncateMode(Enum):
    """
    Specifies how inputs to the NVIDIA embedding components are truncated.
    If START, the input will be truncated from the start.
    If END, the input will be truncated from the end.
    If NONE, an error will be returned.
    """

    START = "START"
    END = "END"
    NONE = "NONE"

    def __str__(self):
        return self.value

    @classmethod
    def _missing_(cls, value: object):
        msg = f"Unknown truncate mode '{value}'. Supported modes are: {list(cls.__members__.keys())}"
        raise ValueError(msg)

    @classmethod
    def from_str(cls, string: str) -> "EmbeddingTruncateMode":
        """
        Create an truncate mode from a string.

        :param string:
            String to convert.
        :returns:
            Truncate mode.
        """
        return cls(string)
