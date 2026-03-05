import logging
import logging.handlers as handlers
import os
from typing import Any, Dict

_LOGGERS: Dict[str, logging.Logger] = {}
_DEFAULT_LOGGER_NAME = "pole"
_LOG_DIR = "./log"
_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
STANDARD_LOG_LEVELS = ("INFO", "WARN", "ERROR")
STANDARD_LOG_KEYWORDS = {
    "APP_START",
    "APP_END",
    "DATA_LOAD",
    "DATA_SAVE",
    "MODEL_SELECT",
    "MODEL_TRAIN",
    "MODEL_EVAL",
    "EVAL_STAGE",
    "MODEL_EXPORT",
    "DB_CONNECT",
    "DB_QUERY",
    "FILE_IO",
}
DEFAULT_LOG_KEYWORD = "GENERAL"


def _normalize_name(name=None) -> str:
    if name is None or str(name).strip() == "":
        return _DEFAULT_LOGGER_NAME
    return str(name).strip()


def make_logger(name=None):
    logger_name = _normalize_name(name)
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger

    os.makedirs(_LOG_DIR, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler()
    file_handler = handlers.TimedRotatingFileHandler(
        filename=os.path.join(_LOG_DIR, f"{logger_name}.log"),
        when="D",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )

    console.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def get_logger(name=None):
    logger_name = _normalize_name(name)
    if logger_name not in _LOGGERS:
        _LOGGERS[logger_name] = make_logger(logger_name)
    return _LOGGERS[logger_name]


def normalize_level(level: str) -> str:
    text = (level or "INFO").upper().strip()
    if text == "WARNING":
        text = "WARN"
    return text if text in STANDARD_LOG_LEVELS else "INFO"


def normalize_keyword(keyword: str) -> str:
    text = (keyword or DEFAULT_LOG_KEYWORD).upper().strip()
    if not text:
        return DEFAULT_LOG_KEYWORD
    return text if text in STANDARD_LOG_KEYWORDS else DEFAULT_LOG_KEYWORD


def format_event_message(keyword: str, message: str, **fields: Any) -> str:
    normalized_keyword = normalize_keyword(keyword)
    base = f"[{normalized_keyword}] {message}"
    if not fields:
        return base
    field_text = " ".join(f"{k}={v}" for k, v in sorted(fields.items()))
    return f"{base} | {field_text}"


def log_event(logger_obj: logging.Logger, level: str, keyword: str, message: str, **fields: Any) -> None:
    normalized_level = normalize_level(level)
    event_message = format_event_message(keyword, message, **fields)
    if normalized_level == "ERROR":
        logger_obj.error(event_message)
    elif normalized_level == "WARN":
        logger_obj.warning(event_message)
    else:
        logger_obj.info(event_message)
