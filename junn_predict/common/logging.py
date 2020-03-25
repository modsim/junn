import logging

import tqdm
import colorlog


class TqdmLoggingHandler(logging.StreamHandler):
    """
    TqdmLoggingHandler
    """
    def emit(self, record):
        message = self.format(record)
        tqdm.tqdm.write(message)


class DelayedFileLog(logging.StreamHandler):
    """
    DelayedFileLog is a log handler which will cache log messages up to the point where the desired log
    filename is set using setFilename, at which
    """
    def __init__(self):
        super().__init__()

        self.file_name = None
        self.buffer = []

    def emit(self, record):
        if self.file_name is None:
            message = self.format(record)
            self.buffer.append(message)
        else:
            super().emit(record)

    def setFilename(self, file_name, mode='a'):
        self.file_name = file_name

        stream = open(file_name, mode)
        for old_message in self.buffer:
            stream.write(old_message + self.terminator)

        self.setStream(stream)


def setup_logging(level):
    name_to_log_level = get_name_to_log_level_dict()
    if level in name_to_log_level:
        level = name_to_log_level[level]

    tqdm_log_handler = TqdmLoggingHandler()
    log_format = "%(asctime)-15s.%(msecs)03d %(process)d %(levelname)s %(name)s %(message)s"
    log_datefmt = '%Y-%m-%d %H:%M:%S'
    tqdm_log_handler.setFormatter(
        colorlog.ColoredFormatter(fmt='%(log_color)s' + log_format, datefmt=log_datefmt)
    )
    buffer = DelayedFileLog()
    log_handlers = [
        tqdm_log_handler,
        buffer
    ]
    # noinspection PyArgumentList
    logging.basicConfig(level=level, format=log_format,
                        datefmt=log_datefmt,
                        handlers=log_handlers)


def get_name_to_log_level_dict():
    # noinspection PyProtectedMember
    name_to_log_level = logging._nameToLevel.copy()
    return name_to_log_level


def get_log_levels():
    log_levels = [k for k, v in sorted(get_name_to_log_level_dict().items(), key=lambda ab: ab[1])]
    log_levels.remove('NOTSET')
    return log_levels
