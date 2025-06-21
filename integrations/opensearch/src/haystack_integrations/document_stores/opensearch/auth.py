from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional, Type, TypeVar

from haystack import default_from_dict, default_to_dict
from haystack.document_stores.errors import DocumentStoreError
from haystack.lazy_imports import LazyImport
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from opensearchpy import AWSV4SignerAsyncAuth, Urllib3AWSV4SignerAuth

with LazyImport("Run 'pip install \"boto3\"' to install boto3.") as boto3_import:
    import boto3
    from botocore.exceptions import BotoCoreError

TSignerAuth = TypeVar("TSignerAuth", Urllib3AWSV4SignerAuth, AWSV4SignerAsyncAuth)


AWS_CONFIGURATION_KEYS = [
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_region_name",
    "aws_profile_name",
]


class AWSConfigurationError(DocumentStoreError):
    """Exception raised when AWS is not configured correctly"""


def _resolve_secret(secret: Optional[Secret]) -> Optional[str]:
    return secret.resolve_value() if secret else None


def _get_aws_session(
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
    aws_region_name: Optional[str] = None,
    aws_profile_name: Optional[str] = None,
    **kwargs: Any,
) -> "boto3.Session":
    """
    Creates an AWS Session with the given parameters.
    Checks if the provided AWS credentials are valid and can be used to connect to AWS.

    :param aws_access_key_id: AWS access key ID.
    :param aws_secret_access_key: AWS secret access key.
    :param aws_session_token: AWS session token.
    :param aws_region_name: AWS region name.
    :param aws_profile_name: AWS profile name.
    :param kwargs: The kwargs passed down to the service client. Supported kwargs depend on the model chosen.
        See https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html.
    :raises AWSConfigurationError: If the provided AWS credentials are invalid.
    :returns: The created AWS session.
    """
    boto3_import.check()
    try:
        return boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=aws_region_name,
            profile_name=aws_profile_name,
        )
    except BotoCoreError as e:
        provided_aws_config = {k: v for k, v in kwargs.items() if k in AWS_CONFIGURATION_KEYS}
        msg = f"Failed to initialize the session with provided AWS credentials {provided_aws_config}"
        raise AWSConfigurationError(msg) from e


@dataclass
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
        self._urllib3_aws_v4_signer_auth = self._get_aws_v4_signer_auth(Urllib3AWSV4SignerAuth)

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
        :return:
            A dictionary containing the signed headers.
        """
        return self._urllib3_aws_v4_signer_auth(method, url, body)

    def _get_aws_v4_signer_auth(self, signer_auth_class: Type[TSignerAuth]) -> TSignerAuth:
        try:
            region_name = _resolve_secret(self.aws_region_name)
            session = _get_aws_session(
                aws_access_key_id=_resolve_secret(self.aws_access_key_id),
                aws_secret_access_key=_resolve_secret(self.aws_secret_access_key),
                aws_session_token=_resolve_secret(self.aws_session_token),
                aws_region_name=region_name,
                aws_profile_name=_resolve_secret(self.aws_profile_name),
            )
            credentials = session.get_credentials()
            return signer_auth_class(
                credentials=credentials,
                # if region is None, the underlying classes raise a clear error at initialization time
                region=region_name,  # type: ignore[arg-type]
                service=self.aws_service,
            )
        except Exception as exception:
            msg = (
                "Could not connect to AWS OpenSearch. Make sure the AWS environment is configured correctly. "
                "See https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration"
            )
            raise AWSConfigurationError(msg) from exception


class AsyncAWSAuth:
    """
    Adapter for using `AWSAuth` with `AsyncOpenSearch`.

    This class works as an adapter for the `AWSAuth` class to be used with the `AsyncOpenSearch` class.
    It facilitates the creation of the `AWSV4SignerAsyncAuth` by making use of Haystack secrets and taking care of
    the necessary `AWSV4SignerAsyncAuth` creation steps including boto3 Sessions and boto3 credentials.

    Note that `AsyncHttpConnection` must be used as `connection_class` with `AsyncOpenSearch` to use this adapter.
    """

    def __init__(self, aws_auth: AWSAuth) -> None:
        """
        Initializes an AsyncAWSAuth instance

        :param aws_auth: The AWSAuth instance to be used for signing requests.
        """
        self.aws_auth = aws_auth
        self._async_aws_v4_signer_auth = self.aws_auth._get_aws_v4_signer_auth(AWSV4SignerAsyncAuth)

    def __call__(self, method: str, url: str, query_string: str, body: Any) -> Dict[str, str]:
        """
        Signs the request and returns headers.

        This method is executed by AsyncHttpConnection when making a request to the OpenSearch service.

        :param method: HTTP method
        :param url: URL
        :param query_string: Query string
        :param body: Body
        :return:
            A dictionary containing the signed headers.
        """
        return self._async_aws_v4_signer_auth(method, url, query_string, body)
