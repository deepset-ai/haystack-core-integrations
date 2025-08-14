# Haystack Downloader Integration

A Haystack integration component for downloading files from multiple storage backends including HTTP/HTTPS, S3, and local filesystem.

## Features

- **Multi-backend support**: HTTP/HTTPS, S3, and local filesystem
- **Automatic backend detection**: URL scheme-based routing
- **Consistent metadata**: Standardized metadata structure across all backends
- **Credential management**: Secure handling of AWS credentials and HTTP authentication
- **Custom backends**: Extensible architecture for custom storage implementations
- **Warm-up functionality**: Early validation of credentials and connections
- **Security**: Path validation and secure credential handling

## Installation

```bash
pip install downloader-haystack
```

## Quick Start

### Basic Usage

```python
from haystack_integrations.components.downloader import Downloader

# Initialize with default backends
downloader = Downloader()

# Download from different sources
result = downloader.run("https://example.com/document.pdf")
print(f"Downloaded {result['metadata']['filename']} ({result['metadata']['size']} bytes)")

result = downloader.run("s3://my-bucket/document.txt")
print(f"Downloaded from S3: {result['metadata']['filename']}")

result = downloader.run("file:///path/to/local/file.txt")
print(f"Downloaded local file: {result['metadata']['filename']}")
```

### With Credentials

```python
from haystack import Secret
from haystack_integrations.components.downloader import Downloader

downloader = Downloader(
    aws_access_key_id=Secret.from_token("your_aws_key"),
    aws_secret_access_key=Secret.from_token("your_aws_secret"),
    aws_region="us-west-2",
    http_auth_token=Secret.from_token("your_bearer_token")
)

# Warm up to validate credentials
downloader.warm_up()
```

### Pipeline Integration

```python
from haystack import Pipeline
from haystack_integrations.components.downloader import Downloader
from haystack.components.converters import TextFileToDocument

# Create pipeline
pipeline = Pipeline()
pipeline.add_component("downloader", Downloader())
pipeline.add_component("converter", TextFileToDocument())

# Connect components
pipeline.connect("downloader.content", "converter.sources")

# Run pipeline
result = pipeline.run({
    "downloader": {"url": "https://example.com/document.txt"}
})

# Access results
document = result["converter"]["documents"][0]
```

## Storage Backends

### HTTP/HTTPS Backend

Handles `http://` and `https://` URLs with support for:
- Basic authentication
- Bearer token authentication
- Custom headers
- SSL verification control
- Redirect handling

```python
downloader = Downloader(
    http_auth_token=Secret.from_token("your_token"),
    http_username=Secret.from_token("username"),
    http_password=Secret.from_token("password"),
    http_headers={"User-Agent": "MyApp/1.0"},
    http_timeout=60.0
)
```

### S3 Backend

Handles `s3://bucket/key` URLs with support for:
- AWS credentials (access key, secret key, session token)
- Region configuration
- Custom S3 endpoints
- SSL verification control

```python
downloader = Downloader(
    aws_access_key_id=Secret.from_token("your_key"),
    aws_secret_access_key=Secret.from_token("your_secret"),
    aws_region="us-west-2",
    s3_endpoint_url="https://custom-s3-endpoint.com"
)
```

### Local Filesystem Backend

Handles `file:///path/to/file` URLs with:
- Path validation and security checks
- File permission handling
- Comprehensive file metadata

```python
downloader = Downloader(
    local_base_path="/allowed/directory"  # Restrict access to specific directory
)
```

## Custom Storage Backends

You can implement custom storage backends by implementing the `StorageBackend` protocol:

```python
from haystack_integrations.components.downloader import Downloader, StorageBackend
from typing import Any, Dict, Tuple

class MyCustomBackend(StorageBackend):
    def download(self, url: str) -> Tuple[bytes, Dict[str, Any]]:
        # Custom download logic
        content = b"custom file content"
        metadata = {"backend": "custom", "filename": "custom_file"}
        return content, metadata
    
    def can_handle(self, url: str) -> bool:
        return url.startswith("custom://")
    
    def warm_up(self) -> None:
        # Custom warm-up logic
        pass

# Use custom backend
custom_backend = MyCustomBackend()
downloader = Downloader(storage_backend=custom_backend)
```

## Metadata Structure

All backends return consistent metadata:

```python
{
    "filename": "document.pdf",           # Extracted filename
    "content_type": "application/pdf",    # MIME type
    "size": 1024,                        # File size in bytes
    "source_url": "https://...",         # Original URL
    "last_modified": "2024-01-01T...",   # Last modified timestamp
    "etag": "abc123",                    # ETag/checksum
    "backend": "http",                   # Backend name
    "download_time": "2024-01-01T...",   # Download timestamp
    "checksum": "md5_hash",              # File checksum
    "headers": {...},                     # HTTP headers (if applicable)
    # Backend-specific fields
    "s3_bucket": "bucket",               # S3-specific
    "local_path": "/path/to/file",       # Local-specific
}
```

## Environment Variables

The component automatically detects standard environment variables:

- **AWS**: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`
- **HTTP**: `HTTP_AUTH_TOKEN`, `HTTP_USERNAME`, `HTTP_PASSWORD`

## Warm-up Functionality

Use the `warm_up()` method to validate credentials and connections early:

```python
downloader = Downloader()
downloader.warm_up()  # Validates all backends

# Now safe to use in pipeline
result = downloader.run("https://example.com/file.txt")
```

## Error Handling

The component provides clear error messages for common issues:

- **File not found**: `FileNotFoundError` with descriptive message
- **Permission denied**: `PermissionError` for access issues
- **Invalid URLs**: `ValueError` for unsupported schemes
- **Authentication failures**: Backend-specific exceptions

## Security Considerations

- Local filesystem access is restricted to the configured base path
- Credentials are handled securely using Haystack's `Secret` utility
- SSL verification is enabled by default for all backends
- Path traversal attacks are prevented through path validation

## Future Enhancements

Planned features for future versions:
- Google Cloud Storage backend
- Azure Blob Storage backend
- Streaming downloads for large files
- Caching and retry mechanisms
- Progress callbacks for long downloads

## Contributing

Contributions are welcome! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE.txt) file for details.
