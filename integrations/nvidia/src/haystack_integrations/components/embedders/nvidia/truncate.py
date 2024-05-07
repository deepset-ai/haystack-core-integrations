from enum import Enum


class TruncateMode(Enum):
    """
    Enum used to specify truncation mode for embeddings.
    """

    START = "START"
    END = "END"

    def __str__(self):
        return self.value

    @classmethod
    def from_str(cls, string: str) -> "TruncateMode":
        """
        Create an truncate mode from a string.

        :param string:
            String to convert.
        :returns:
            Truncate mode.
        """
        enum_map = {e.value: e for e in TruncateMode}
        opt_mode = enum_map.get(string)
        if opt_mode is None:
            msg = f"Unknown truncate mode '{string}'. Supported modes are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return opt_mode
