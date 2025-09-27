# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0


class S3Error(Exception):
    """Exception for issues that occur in the S3 based components"""


class S3ConfigurationError(S3Error):
    """Exception raised when AmazonS3 node is not configured correctly"""


class S3StorageError(S3Error):
    """This exception is raised when an error occurs while interacting with a S3Storage object."""
