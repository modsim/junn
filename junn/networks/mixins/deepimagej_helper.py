"""Mixin helper to create DeepImageJ compatible metadata."""
import datetime
import os
from functools import reduce

import numpy as np
from tifffile import imsave as tifffile_imsave

from ...common import distributed


def serialize_value(value):
    """
    Serialize a value in an DeepImageJ XML compatible manner.

    :param value:
    :return:
    """
    if isinstance(value, bool):
        return str(value).lower()
    else:
        return str(value)


def serialize_xml(the_dict):
    """
    Serialize a dictionary into a XML structure.

    :param the_dict:
    :return:
    """

    def _inner(inner_dict, level=0):
        return ''.join(
            reduce(
                lambda a, b: a + b,
                (
                    (
                        [('\t' * level) + '<%s>' % (key,) + '\n']
                        + [_inner(value, level + 1)]
                        + [('\t' * level) + '</%s>' % (key,) + '\n']
                    )
                    if isinstance(value, dict)
                    else (
                        [
                            ('\t' * level)
                            + '<%s>%s</%s>'
                            % (
                                key,
                                serialize_value(value),
                                key,
                            )
                            + '\n'
                        ]
                    )
                    for key, value in inner_dict.items()
                ),
            )
        )

    return '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + _inner(the_dict)


def write_to(file_name, data):
    """
    Write text to a file.

    :param file_name:
    :param data:
    :return:
    """
    with open(file_name, 'w+') as fp:
        fp.write(data)


class DeepImageJCompatibilityMixin:
    """Mixin to give DeepImageJ metadata writing."""

    def save_model(self):  # noqa: D102
        if not distributed.is_rank_zero():  # only rank zero may save
            return

        super().save_model()

        # signature_to_export = self.signatures[self.PREDICTION_SIGNATURE]

        # name_input = signature_to_export.input_signature[0].name
        # name_output = signature_to_export.get_concrete_function(
        # ).outputs[0].name  # no?

        now = datetime.datetime.now().isoformat()
        name = os.path.basename(self.model_path)
        base = {
            'Model': {
                'ModelInformation': {
                    'Name': name,
                    'Author': 'A JUNN User',
                    'URL': 'https://modsim.github.io/junn',
                    'Credit': 'A JUNN User',
                    'Version': 1,
                    'Date': now,
                    'Reference': 'Personal communication.',
                },
                'ModelTest': {  # dummy values
                    'InputSize': '1024x1024',
                    'OutputSize': '1024x1024',
                    'MemoryPeak': '1024.0 Mb',
                    'Runtime': '1.0 s',
                    'PixelSize': '1.0µmx1.0µm',
                },
                'ModelCharacteristics': {
                    'ModelTag': 'tf.saved_model.tag_constants.SERVING',
                    'SignatureDefinition': 'predict',
                    'InputTensorDimensions': ',-1,-1,-1,',
                    'NumberOfInputs': 1,
                    'InputNames0': 'image',
                    'InputOrganization0': 'HWC',
                    'NumberOfOutputs': 1,
                    'OutputNames0': 'output_0',
                    'OutputOrganization0': 'HWC',
                    'Channels': 1,  # does this have to be dynamic?
                    'FixedPatch': False,
                    'MinimumSize': 1,
                    'PatchSize': 1024,
                    'FixedPadding': False,
                    'Padding': 1,
                    'PreprocessingFile': 'preprocessing.txt',
                    'PostprocessingFile': 'postprocessing.txt',
                    'slices': 1,
                },
            }
        }

        configuration = base  # update it afterwards

        def p(file_name):
            return os.path.join(self.model_path, file_name)

        # write necessary files
        write_to(p('config.xml'), serialize_xml(configuration))

        write_to(p('preprocessing.txt'), 'run("32-bit");')

        write_to(p('postprocessing.txt'), '')

        tifffile_imsave(
            p('exampleImage.tiff'),
            np.zeros(
                (
                    32,
                    32,
                ),
                dtype=np.float32,
            ),
        )

        tifffile_imsave(
            p('resultImage.tiff'),
            np.zeros(
                (
                    32,
                    32,
                ),
                dtype=np.float32,
            ),
        )
