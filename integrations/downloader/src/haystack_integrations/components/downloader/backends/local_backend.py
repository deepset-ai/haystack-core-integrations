# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from haystack import logging

from .base import StorageBackend

logger = logging.getLogger(__name__)


class LocalBackend(StorageBackend):
    """
    Local filesystem storage backend for downloading files from the local filesystem.

    Handles file:// URLs and provides comprehensive metadata including file attributes,
    permissions, and checksums. Ensures secure file access with proper path validation.
    """

    def __init__(self, base_path: str = "/"):
        """
        Initialize the local backend.

        :param base_path: Base directory path for security (default: /)
        """
        self.base_path = Path(base_path).resolve()
        logger.debug(f"Local backend initialized with base path: {self.base_path}")

    def can_handle(self, url: str) -> bool:
        """
        Check if this backend can handle the given URL.

        :param url: The URL to check
        :return: True if the URL starts with file://
        """
        return url.startswith("file://")

    def download(self, url: str) -> tuple[bytes, dict[str, Any]]:
        """
        Download file from local filesystem URL.

        :param url: The file:// URL to download from
        :return: Tuple of (file_content: bytes, metadata: Dict[str, Any])
        :raises: ValueError, FileNotFoundError, PermissionError, OSError
        """
        if not self.can_handle(url):
            error_msg = f"Local backend cannot handle URL: {url}"
            raise ValueError(error_msg)

        # Parse file URL
        file_path = self._parse_file_url(url)
        if not file_path:
            error_msg = f"Invalid file URL format: {url}"
            raise ValueError(error_msg)

        # Validate file path for security
        self._validate_file_path(file_path)

        logger.debug(f"Downloading file from local filesystem: {file_path}")

        try:
            # Read file content
            with open(file_path, "rb") as f:
                content = f.read()

            # Extract metadata
            metadata = self._extract_metadata(url, file_path, content)

            logger.debug(f"Successfully downloaded file from {file_path}, size: {len(content)} bytes")
            return content, metadata

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except PermissionError:
            logger.error(f"Permission denied accessing file: {file_path}")
            raise
        except OSError as e:
            logger.error(f"OS error reading file {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading file {file_path}: {e}")
            raise

    def _parse_file_url(self, url: str) -> str:
        """
        Parse file:// URL to extract file path.

        :param url: File URL in format file:///path/to/file
        :return: File path string
        """
        parsed = urlparse(url)
        if parsed.scheme != "file":
            return ""

        # Handle both file:///path and file://path formats
        if parsed.netloc:
            # file://host/path format (rare, but possible)
            path = f"/{parsed.netloc}{parsed.path}"
        else:
            # file:///path format (standard)
            path = parsed.path

        return path

    def _validate_file_path(self, file_path: str) -> None:
        """
        Validate file path for security and accessibility.

        :param file_path: The file path to validate
        :raises: ValueError, PermissionError
        """
        path = Path(file_path).resolve()

        # Check if path is within base path (security check)
        try:
            path.relative_to(self.base_path)
        except ValueError:
            error_msg = f"File path {file_path} is outside allowed base path {self.base_path}"
            raise ValueError(error_msg) from None

        # Check if file exists
        if not path.exists():
            error_msg = f"File not found: {file_path}"
            raise FileNotFoundError(error_msg)

        # Check if it's a file (not directory)
        if not path.is_file():
            error_msg = f"Path is not a file: {file_path}"
            raise ValueError(error_msg)

        # Check if file is readable
        if not os.access(path, os.R_OK):
            error_msg = f"File is not readable: {file_path}"
            raise PermissionError(error_msg)

    def _extract_metadata(self, url: str, file_path: str, content: bytes) -> dict[str, Any]:
        """
        Extract metadata from local file.

        :param url: The original file:// URL
        :param file_path: The local file path
        :param content: File content
        :return: Dictionary containing metadata
        """
        path = Path(file_path)

        # Get file stats
        stat_info = path.stat()

        # Extract filename
        filename = path.name

        # Determine content type based on file extension
        content_type = self._get_content_type(path)

        # Get file size
        size = stat_info.st_size

        # Get last modified time
        last_modified = datetime.fromtimestamp(stat_info.st_mtime, tz=timezone.utc).isoformat()

        # Get file permissions
        permissions = stat.filemode(stat_info.st_mode)

        # Calculate MD5 checksum
        checksum = hashlib.md5(content).hexdigest()

        # Get file owner and group (if available)
        try:
            owner = path.owner()
        except (OSError, KeyError):
            owner = None

        try:
            group = path.group()
        except (OSError, KeyError):
            group = None

        # Build metadata
        metadata = {
            "filename": filename,
            "content_type": content_type,
            "size": size,
            "source_url": url,
            "last_modified": last_modified,
            "etag": None,  # Local files don't have ETags
            "backend": "local",
            "download_time": datetime.now(timezone.utc).isoformat(),
            "checksum": checksum,
            "headers": {},  # Local files don't have HTTP headers
            "local_path": str(file_path),
            "local_permissions": permissions,
            "local_owner": owner,
            "local_group": group,
            "local_inode": stat_info.st_ino,
            "local_device": stat_info.st_dev,
        }

        return metadata

    def _get_content_type(self, path: Path) -> str:
        """
        Determine content type based on file extension.

        :param path: File path
        :return: MIME type string
        """
        extension = path.suffix.lower()

        # Common file extensions and their MIME types
        mime_types = {
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".html": "text/html",
            ".htm": "text/html",
            ".css": "text/css",
            ".js": "application/javascript",
            ".json": "application/json",
            ".xml": "application/xml",
            ".pdf": "application/pdf",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".xls": "application/vnd.ms-excel",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".ppt": "application/vnd.ms-powerpoint",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".zip": "application/zip",
            ".tar": "application/x-tar",
            ".gz": "application/gzip",
            ".7z": "application/x-7z-compressed",
            ".rar": "application/vnd.rar",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".svg": "image/svg+xml",
            ".mp3": "audio/mpeg",
            ".mp4": "video/mp4",
            ".avi": "video/x-msvideo",
            ".mov": "video/quicktime",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
        }

        return mime_types.get(extension, "application/octet-stream")

    def warm_up(self) -> None:
        """
        Test local filesystem access.

        Tests by accessing a temporary directory to validate filesystem permissions.
        Logs warnings if warm-up fails but doesn't raise exceptions.
        """
        try:
            # Test by accessing a temporary directory
            with tempfile.NamedTemporaryFile() as _:
                pass
            logger.info("Local filesystem backend warmed up successfully")
        except Exception as e:
            logger.warning(f"Local filesystem backend warm-up failed: {e}")
