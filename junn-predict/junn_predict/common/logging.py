"""Logging helpers."""
import logging
import sys

import colorlog
import tqdm


class TqdmLoggingHandler(logging.StreamHandler):
    """TqdmLoggingHandler, outputs log messages to the console compatible with tqdm."""

    def emit(self, record):  # noqa: D102
        message = self.format(record)
        tqdm.tqdm.write(message)


class DelayedFileLog(logging.StreamHandler):
    """DelayedFileLog will cache messages till it can write them to a specified file."""

    def __init__(self):  # noqa: D107
        super().__init__()

        self.file_name = None
        self.buffer = []

    def emit(self, record):  # noqa: D102
        if self.file_name is None:
            message = self.format(record)
            self.buffer.append(message)
        else:
            super().emit(record)

    def setFilename(self, file_name, mode='a'):
        """
        Set the filename to write the log messages to.

        :param file_name: File name to use.
        :param mode: File open mode, by default 'a'.
        :return: None
        """
        self.file_name = file_name

        stream = open(file_name, mode)
        for old_message in self.buffer:
            stream.write(old_message + self.terminator)

        self.setStream(stream)


def setup_logging(level):
    """
    Set the logging up to the specified level.

    :param level: Log level
    :return: None
    """
    name_to_log_level = get_name_to_log_level_dict()
    if level in name_to_log_level:
        level = name_to_log_level[level]

    tqdm_log_handler = TqdmLoggingHandler()
    log_format = (
        "%(asctime)-15s.%(msecs)03d %(process)d %(levelname)s %(name)s %(message)s"
    )
    log_datefmt = '%Y-%m-%d %H:%M:%S'
    tqdm_log_handler.setFormatter(
        colorlog.TTYColoredFormatter(
            fmt='%(log_color)s' + log_format, datefmt=log_datefmt, stream=sys.stdout
        )
    )
    buffer = DelayedFileLog()
    log_handlers = [tqdm_log_handler, buffer]
    # noinspection PyArgumentList
    logging.basicConfig(
        level=level, format=log_format, datefmt=log_datefmt, handlers=log_handlers
    )


def get_name_to_log_level_dict():
    """
    Return a dict with a mapping of log levels.

    :return: The dict
    """
    # noinspection PyProtectedMember
    name_to_log_level = logging._nameToLevel.copy()
    return name_to_log_level


def get_log_levels():
    """
    Return supported log levels.

    :return: List of log levels
    """
    log_levels = [
        k for k, v in sorted(get_name_to_log_level_dict().items(), key=lambda ab: ab[1])
    ]
    log_levels.remove('NOTSET')
    return log_levels
