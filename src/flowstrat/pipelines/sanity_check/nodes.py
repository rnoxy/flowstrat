import logging

logger = logging.getLogger(__name__)


def sanity_check(params: dict[str, any]):
    logger.info("params={}".format(params))
    return dict()
