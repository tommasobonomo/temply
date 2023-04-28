import logging


def logger_factory(
    module_name: str = "dee", level: int = logging.WARNING
) -> logging.Logger:
    # Initialise logger
    logger = logging.getLogger(module_name)
    logger.propagate = False
    logger.setLevel(level)

    # Init console handler and add to logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s: %(levelname)s] %(message)s")
    )
    logger.addHandler(console_handler)

    return logger
