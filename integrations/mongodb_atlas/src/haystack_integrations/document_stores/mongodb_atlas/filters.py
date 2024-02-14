import logging

logger = logging.getLogger(__name__)


def haystack_filters_to_mongo(_):
    # TODO
    logger.warning("Filtering not yet implemented for MongoDBAtlasDocumentStore")
    return {}
