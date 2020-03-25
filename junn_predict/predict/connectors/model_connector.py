from tunable import Selectable


class ModelConnector(Selectable):
    def __init__(self, arg):
        pass

    def get_signatures(self):
        return []

    def call(self, signature, data):
        pass

    @staticmethod
    def check_import():
        pass
