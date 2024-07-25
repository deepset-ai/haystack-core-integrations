from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional, Union

from haystack.utils.auth import Secret, deserialize_secrets_inplace


@dataclass(frozen=True)
class AWSAuth:
    """
    Auth credentials for AWS OpenSearch services.
    """

    aws_access_key_id: Optional[Secret] = field(
        default_factory=lambda: Secret.from_env_var("AWS_ACCESS_KEY_ID", strict=False)
    )
    aws_secret_access_key: Optional[Secret] = field(
        default_factory=lambda: Secret.from_env_var("AWS_SECRET_ACCESS_KEY", strict=False)
    )
    aws_session_token: Optional[Secret] = field(
        default_factory=lambda: Secret.from_env_var("AWS_SESSION_TOKEN", strict=False)
    )
    aws_region_name: Optional[Secret] = field(
        default_factory=lambda: Secret.from_env_var("AWS_DEFAULT_REGION", strict=False)
    )
    aws_profile_name: Optional[Secret] = field(default_factory=lambda: Secret.from_env_var("AWS_PROFILE", strict=False))
    aws_service: str = field(default="es")

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the object to a dictionary representation for serialization.
        """
        _fields = {}
        for _field in fields(self):
            field_value = getattr(self, _field.name)
            if _field.type == Optional[Secret]:
                _fields[_field.name] = field_value.to_dict() if field_value is not None else None
            else:
                _fields[_field.name] = field_value

        return {**_fields}

    @classmethod
    def from_dict(cls, data: Union[Dict[str, Any], bool]) -> Optional["AWSAuth"]:
        """
        Converts a dictionary representation to an AWSAuth object.
        """
        if isinstance(data, bool):
            return cls() if data else None

        deserialize_secrets_inplace(
            data,
            ["aws_access_key_id", "aws_secret_access_key", "aws_session_token", "aws_region_name", "aws_profile_name"],
        )
        return cls(**data)
