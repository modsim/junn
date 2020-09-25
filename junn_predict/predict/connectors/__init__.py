import sys

from urllib.parse import urlparse

from .grpc_connector import GRPCConnector
from .http_connector import HTTPConnector
from .local_model import LocalModel


def suggest_connector(arg):
    urlfragments = urlparse(arg)

    if sys.platform.startswith('win'):
        if len(urlfragments.scheme) == 1:  # likely a drive letter
            return LocalModel

    try:
        return schema_model_mapping[urlfragments.scheme]
    except KeyError:
        return None


schema_model_mapping = {
    '': LocalModel,
    'file': LocalModel,
    'grpc': GRPCConnector,
    'grpcs': GRPCConnector,
    'tfs+http': HTTPConnector,
    'tfs+https': HTTPConnector,
    'http': LocalModel,
    'https': LocalModel,
}
