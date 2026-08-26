import logging

logger = logging.getLogger(__name__)


def calculate_mean(values: list[float]) -> float:
    logger.info("Calculating mean for %d values", len(values))

    if not values:
        logger.error("Cannot calculate mean of an empty list")
        raise ValueError("Values cannot be empty.")

    return sum(values) / len(values)
