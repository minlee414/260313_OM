import logging
from pathlib import Path


def setupLogger(logPath: Path, logLevel: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("etched_om_microstructure_analyzer")
    logger.setLevel(getattr(logging, logLevel.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    fileHandler = logging.FileHandler(logPath, encoding="utf-8")
    fileHandler.setFormatter(formatter)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    return logger