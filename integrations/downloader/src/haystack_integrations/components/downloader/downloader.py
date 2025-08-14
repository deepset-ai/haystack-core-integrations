# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret

from .backends import HTTPBackend, LocalBackend, S3Backend, StorageBackend

logger = logging.getLogger(__name__)


@component
class Downloader:
    """
    Downloader component for fetching files from multiple storage backends.

    Supports HTTP/HTTPS, S3, and local filesystem storage with automatic
    backend detection based on URL scheme. Provides consistent metadata
    structure across all backends and supports custom storage backends.

    ### Usage Examples

    #### Basic Usage with Environment Variables
    ```python
    from haystack_integrations.components.downloader import Downloader

    # Uses environment variables for credentials
    downloader = Downloader()

    # Warm up before using (optional but recommended)
    downloader.warm_up()

    # Download from different sources
    result = downloader.run("https://example.com/file.pdf")
    result = downloader.run("s3://my-bucket/document.txt")
    result = downloader.run("file:///path/to/local/file.txt")
    ```

    #### Explicit Credential Configuration
    ```python
    from haystack import Secret
    from haystack_integrations.components.downloader import Downloader

    downloader = Downloader(
        aws_access_key_id=Secret.from_token("my_aws_key"),
        aws_secret_access_key=Secret.from_token("my_aws_secret"),
        aws_region="us-west-2",
        http_auth_token=Secret.from_token("my_bearer_token")
    )

    # Warm up to validate all credentials
    downloader.warm_up()
    ```

    #### Custom Storage Backend
    ```python
    from haystack_integrations.components.downloader import Downloader, StorageBackend

    class MyCustomBackend(StorageBackend):
        def download(self, url: str) -> Tuple[bytes, Dict[str, Any]]:
            # Custom download logic
            pass

        def can_handle(self, url: str) -> bool:
            return url.startswith("custom://")

        def warm_up(self) -> None:
            # Custom warm-up logic
            pass

    custom_backend = MyCustomBackend()
    downloader = Downloader(storage_backend=custom_backend)

    # Warm up custom backend
    downloader.warm_up()
    ```
    """

    def __init__(
        self,
        storage_backend: Optional[StorageBackend] = None,
        # HTTP backend credentials
        http_auth_token: Optional[Secret] = None,
        http_username: Optional[Secret] = None,
        http_password: Optional[Secret] = None,
        http_headers: Optional[dict[str, str]] = None,
        http_timeout: float = 30.0,
        http_max_redirects: int = 5,
        http_verify_ssl: bool = True,
        # S3 backend credentials
        aws_access_key_id: Optional[Secret] = None,
        aws_secret_access_key: Optional[Secret] = None,
        aws_region: Optional[str] = None,
        aws_session_token: Optional[Secret] = None,
        s3_endpoint_url: Optional[str] = None,
        s3_verify_ssl: bool = True,
        # Local backend configuration
        local_base_path: str = "/",
        **backend_kwargs: Any,
    ):
        """
        Initialize the Downloader component.

        :param storage_backend: Optional custom storage backend implementation
        :param http_auth_token: Bearer token for HTTP authentication
        :param http_username: Username for HTTP basic authentication
        :param http_password: Password for HTTP basic authentication
        :param http_headers: Additional HTTP headers
        :param http_timeout: HTTP request timeout in seconds
        :param http_max_redirects: Maximum HTTP redirects to follow
        :param http_verify_ssl: Whether to verify HTTP SSL certificates
        :param aws_access_key_id: AWS access key ID
        :param aws_secret_access_key: AWS secret access key
        :param aws_region: AWS region name
        :param aws_session_token: AWS session token for temporary credentials
        :param s3_endpoint_url: Custom S3 endpoint URL
        :param s3_verify_ssl: Whether to verify S3 SSL certificates
        :param local_base_path: Base directory path for local filesystem access
        :param backend_kwargs: Additional backend-specific parameters
        """
        if storage_backend is None:
            # Initialize default backends with credentials from parameters
            self._init_default_backends(
                http_auth_token=http_auth_token,
                http_username=http_username,
                http_password=http_password,
                http_headers=http_headers,
                http_timeout=http_timeout,
                http_max_redirects=http_max_redirects,
                http_verify_ssl=http_verify_ssl,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_region=aws_region,
                aws_session_token=aws_session_token,
                s3_endpoint_url=s3_endpoint_url,
                s3_verify_ssl=s3_verify_ssl,
                local_base_path=local_base_path,
                **backend_kwargs,
            )
        else:
            self.storage_backend = storage_backend

    def _init_default_backends(
        self,
        http_auth_token: Optional[Secret] = None,
        http_username: Optional[Secret] = None,
        http_password: Optional[Secret] = None,
        http_headers: Optional[dict[str, str]] = None,
        http_timeout: float = 30.0,
        http_max_redirects: int = 5,
        http_verify_ssl: bool = True,
        aws_access_key_id: Optional[Secret] = None,
        aws_secret_access_key: Optional[Secret] = None,
        aws_region: Optional[str] = None,
        aws_session_token: Optional[Secret] = None,
        s3_endpoint_url: Optional[str] = None,
        s3_verify_ssl: bool = True,
        local_base_path: str = "/",
        **backend_kwargs: Any,
    ) -> None:
        """Initialize default storage backends with provided credentials."""
        # Initialize HTTP backend
        self.http_backend = HTTPBackend(
            auth_token=http_auth_token,
            username=http_username,
            password=http_password,
            headers=http_headers,
            timeout=http_timeout,
            max_redirects=http_max_redirects,
            verify_ssl=http_verify_ssl,
        )

        # Initialize S3 backend
        self.s3_backend = S3Backend(
            access_key_id=aws_access_key_id,
            secret_access_key=aws_secret_access_key,
            region=aws_region,
            session_token=aws_session_token,
            endpoint_url=s3_endpoint_url,
            verify_ssl=s3_verify_ssl,
        )

        # Initialize local backend
        self.local_backend = LocalBackend(base_path=local_base_path)

        logger.debug("Initialized default storage backends: HTTP, S3, Local")

    def warm_up(self) -> None:
        """
        Warm up the component by executing authentication and connection tests.

        This method can be called before running the pipeline to ensure all backends
        are properly configured and authenticated. It logs warnings for any failures
        but doesn't prevent the component from being used.
        """
        if hasattr(self, "storage_backend"):
            # Custom backend warm up
            if hasattr(self.storage_backend, "warm_up"):
                try:
                    self.storage_backend.warm_up()
                except Exception as e:
                    logger.warning(f"Custom backend warm-up failed: {e}")
        else:
            # Warm up all default backends
            for backend_name, backend in [
                ("HTTP", self.http_backend),
                ("S3", self.s3_backend),
                ("Local", self.local_backend),
            ]:
                if hasattr(backend, "warm_up"):
                    try:
                        backend.warm_up()
                    except Exception as e:
                        logger.warning(f"{backend_name} backend warm-up failed: {e}")

    @component.output_types(content=bytes, metadata=dict[str, Any])
    def run(self, url: str) -> dict[str, Any]:
        """
        Download file from the given URL and return content + metadata.

        :param url: The URL to download from (supports http://, https://, s3://, file://)
        :return: Dictionary containing 'content' (bytes) and 'metadata' (dict)
        :raises: ValueError, FileNotFoundError, PermissionError, and other backend-specific exceptions
        """
        if not url:
            error_msg = "URL cannot be empty"
            raise ValueError(error_msg)

        logger.debug(f"Downloading file from URL: {url}")

        # Get appropriate backend for the URL
        backend = self._get_backend_for_url(url)

        # Download file using the selected backend
        content, metadata = backend.download(url)

        logger.debug(f"Successfully downloaded file from {url}, size: {len(content)} bytes")

        return {"content": content, "metadata": metadata}

    def _get_backend_for_url(self, url: str) -> StorageBackend:
        """
        Determine which backend to use based on URL scheme.

        :param url: The URL to analyze
        :return: The appropriate storage backend
        :raises: ValueError if no backend can handle the URL
        """
        if hasattr(self, "storage_backend"):
            # Custom backend - check if it can handle the URL
            if self.storage_backend.can_handle(url):
                return self.storage_backend
            else:
                error_msg = f"Custom backend cannot handle URL: {url}"
                raise ValueError(error_msg)

        # Default backends - route based on URL scheme
        if url.startswith("s3://"):
            return self.s3_backend
        elif url.startswith("file://"):
            return self.local_backend
        elif url.startswith(("http://", "https://")):
            return self.http_backend
        else:
            error_msg = f"Unsupported URL scheme: {url}"
            raise ValueError(error_msg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component to a dictionary."""
        if hasattr(self, "storage_backend"):
            # Custom backend - serialize the backend
            return default_to_dict(
                self,
                storage_backend=self.storage_backend,
                **self._get_custom_backend_params(),
            )
        else:
            # Default backends - serialize backend parameters
            return default_to_dict(
                self,
                http_auth_token=self.http_backend.auth_token,
                http_username=self.http_backend.username,
                http_password=self.http_backend.password,
                http_headers=self.http_backend.headers,
                http_timeout=self.http_backend.timeout,
                http_max_redirects=self.http_backend.max_redirects,
                http_verify_ssl=self.http_backend.verify_ssl,
                aws_access_key_id=self.s3_backend.access_key_id,
                aws_secret_access_key=self.s3_backend.secret_access_key,
                aws_region=self.s3_backend.region,
                aws_session_token=self.s3_backend.session_token,
                s3_endpoint_url=self.s3_backend.endpoint_url,
                s3_verify_ssl=self.s3_backend.verify_ssl,
                local_base_path=str(self.local_backend.base_path),
            )

    def _get_custom_backend_params(self) -> dict[str, Any]:
        """Get parameters for custom backend serialization."""
        # This method can be overridden by subclasses to provide custom parameters
        return {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Downloader":
        """Deserialize the component from a dictionary."""
        if "storage_backend" in data:
            # Custom backend - deserialize the backend
            return default_from_dict(cls, data)
        else:
            # Default backends - deserialize backend parameters
            return default_from_dict(cls, data)
