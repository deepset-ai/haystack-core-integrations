from typing import Optional


class MongoDBAtlasDocumentStoreError(Exception):
    """Exception for issues that occur in a MongoDBAtlas document store"""

    def __init__(self, message: Optional[str] = None):
        super().__init__(message=message)
