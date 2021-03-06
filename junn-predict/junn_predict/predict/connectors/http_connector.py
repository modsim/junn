import base64
from io import BytesIO
from urllib.parse import urlparse, urlunparse

from .model_connector import ModelConnector


def assert_pop(frag, what):
    if frag:
        assert frag.pop(0) == what


def return_pop(frag, alternative):
    if frag:
        value = frag.pop(0)
        return alternative if alternative else value


def b64(input_):
    return base64.b64encode(input_).decode('utf8')


def b64d(input_):
    return base64.b64decode(input_)


class HTTPConnector(ModelConnector):
    @staticmethod
    def check_import():
        import numpy as np
        import requests
        from PIL import Image

    def __init__(self, arg, name='', version=''):
        super().__init__(arg)

        url_fragments = urlparse(arg)

        if not url_fragments.scheme:
            # noinspection PyProtectedMember
            url_fragments = url_fragments._replace(scheme='http')

        if 'tfs+' in url_fragments.scheme:
            # noinspection PyProtectedMember
            url_fragments = url_fragments._replace(
                scheme=url_fragments.scheme.replace('tfs+', '')
            )

        path_fragments = url_fragments.path.split('/')

        assert_pop(path_fragments, '')

        assert_pop(path_fragments, 'v1')

        assert_pop(path_fragments, 'models')

        name = return_pop(path_fragments, name)

        assert_pop(path_fragments, 'versions')

        version = return_pop(path_fragments, version)

        if not version:
            version = '1'

        version = str(version)

        # noinspection PyProtectedMember
        url_fragments = url_fragments._replace(
            path='/'.join(['', 'v1', 'models', name, 'versions', version])
        )

        self.url = urlunparse(url_fragments)

    def get_signatures(self):
        import requests

        result = requests.get(self.url + '/metadata')
        result_json = result.json()

        signatures = list(
            sorted(
                name
                for name, signature_def in result_json['metadata']['signature_def'][
                    'signature_def'
                ].items()
                if name != '__saved_model_init_op'
            )
        )

        return signatures

    def call(self, signature, data):
        import numpy as np
        import requests
        from PIL import Image

        def _encode_png(image_data):
            buffer = BytesIO()
            if image_data.ndim == 3 and image_data.shape[2] == 1:
                image_data = image_data[:, :, 0]

            image_data = image_data.astype('uint8')
            Image.fromarray(image_data).save(buffer, format='PNG', compress_level=0)
            return buffer.getvalue()

        def _decode_png(png_data):
            return np.array(Image.open(BytesIO(png_data)))

        if not signature.endswith('_png'):  # a bit of convention here
            inputs = data.tolist()
        else:
            inputs = {'b64': b64(_encode_png(data))}

        data = {'signature_name': signature, 'inputs': inputs}

        result = requests.post(self.url + ':predict', json=data)

        result_json = result.json()

        assert 'error' not in result_json, "Error processing data"

        outputs = result_json['outputs']

        if isinstance(outputs, list):
            outputs = np.array(outputs)
        else:
            assert 'b64' in outputs
            outputs = _decode_png(b64d(outputs['b64']))
        return outputs
