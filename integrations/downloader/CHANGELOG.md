# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Haystack Downloader Integration
- Support for HTTP/HTTPS storage backend with authentication
- Support for S3 storage backend with AWS credentials
- Support for local filesystem storage backend with security validation
- Custom storage backend protocol for extensibility
- Warm-up functionality for early credential validation
- Comprehensive metadata extraction across all backends
- URL scheme-based automatic backend routing
- Secure credential management using Haystack Secret utility
- Pipeline integration support

### Features
- **HTTP/HTTPS Backend**: Basic auth, bearer tokens, custom headers, SSL control
- **S3 Backend**: AWS credentials, region support, custom endpoints, SSL control
- **Local Backend**: Path validation, security checks, comprehensive file metadata
- **Custom Backends**: Protocol-based extensibility for custom storage implementations
- **Metadata Consistency**: Standardized metadata structure across all backends
- **Error Handling**: Clear error messages and proper exception propagation
- **Security**: Path validation, credential security, SSL verification

### Technical Details
- Protocol-based design using Python's typing.Protocol
- Automatic backend detection and routing
- Comprehensive logging and debugging support
- Type hints throughout the codebase
- Follows Haystack integration patterns and conventions
- Comprehensive test coverage planned
- Documentation with usage examples
