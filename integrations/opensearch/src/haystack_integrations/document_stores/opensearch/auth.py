from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional

from haystack import default_from_dict, default_to_dict
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from haystack_integrations.common.opensearch.errors import AWSConfigurationError
from haystack_integrations.common.opensearch.utils import get_aws_session
from opensearchpy import Urllib3AWSV4SignerAuth


@dataclass()
class AWSAuth:
    """
    Auth credentials for AWS OpenSearch services.

    This class works as a thin wrapper around the `Urllib3AWSV4SignerAuth` class from the `opensearch-py` library.
    It facilitates the creation of the `Urllib3AWSV4SignerAuth` by making use of Haystack secrets and taking care of
    the necessary `Urllib3AWSV4SignerAuth` creation steps including boto3 Sessions and boto3 credentials.
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

    def __post_init__(self) -> None:
        """
        Initializes the AWSAuth object.
        """
        self._urllib3_aws_v4_signer_auth = self._get_urllib3_aws_v4_signer_auth()

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

        return default_to_dict(self, **_fields)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional["AWSAuth"]:
        """
        Converts a dictionary representation to an AWSAuth object.
        """
        init_parameters = data.get("init_parameters", {})
        deserialize_secrets_inplace(
            init_parameters,
            ["aws_access_key_id", "aws_secret_access_key", "aws_session_token", "aws_region_name", "aws_profile_name"],
        )
        return default_from_dict(cls, data)

    def __call__(self, method: str, url: str, body: Any) -> Dict[str, str]:
        """
        Signs the request and returns headers.

        This method is executed by Urllib3 when making a request to the OpenSearch service.

        :param method: HTTP method
        :param url: URL
        :param body: Body
        """
        return self._urllib3_aws_v4_signer_auth(method, url, body)

    def _get_urllib3_aws_v4_signer_auth(self) -> Urllib3AWSV4SignerAuth:
        def resolve_secret(secret: Optional[Secret]) -> Optional[str]:
            return secret.resolve_value() if secret else None

        try:
            region_name = resolve_secret(self.aws_region_name)
            session = get_aws_session(
                aws_access_key_id=resolve_secret(self.aws_access_key_id),
                aws_secret_access_key=resolve_secret(self.aws_secret_access_key),
                aws_session_token=resolve_secret(self.aws_session_token),
                aws_region_name=region_name,
                aws_profile_name=resolve_secret(self.aws_profile_name),
            )
            credentials = session.get_credentials()
            return Urllib3AWSV4SignerAuth(credentials, region_name, self.aws_service)
        except Exception as exception:
            msg = (
                "Could not connect to AWS OpenSearch. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise AWSConfigurationError(msg) from exception
