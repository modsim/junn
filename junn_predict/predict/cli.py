import glob
import logging
import os

from .detectors import ROIDetector
from ..common.timed import Timed

from urllib.parse import urlparse, urlunparse

from ..common.cli import get_common_argparser_and_setup
from .connectors import HTTPConnector, suggest_connector
from .connectors.model_connector import ModelConnector

import tqdm
from tifffile import TiffWriter
from roifile import ImagejRoi

from pilyso_io.imagestack import ImageStack, Dimensions, parse_range, prettify_numpy_array
from pilyso_io.imagestack.readers import *

import numpy as np

log = logging.getLogger(__name__)


def check_for_overwriting(inputs, outputs, overwrite=False):
    inputs_outputs = []
    for input_filename, output_filename in zip(inputs, outputs):
        if os.path.isfile(output_filename):

            if overwrite:
                log.info("File \"%s\" already exists, overwriting.", output_filename)
            else:
                continue
        inputs_outputs.append((input_filename, output_filename,))
    return inputs_outputs


def remove_query_if_present(input_filename):
    input_filename_to_use = input_filename
    if '?' in input_filename_to_use:
        input_filename_to_use = urlunparse(
            urlparse(input_filename_to_use)._replace(params='', query='', fragment='', scheme='', netloc='')
        )
    return input_filename_to_use


OUTPUT_WILDCARD = '{}'


def prepare_inputs_outputs(args):
    inputs, output = args.input, args.output

    log.info("Checking inputs (%r) and outputs (%r)", inputs, output)

    if len(inputs) == 1:
        intermediate_input = inputs[0]

        if not ('://' in intermediate_input and 'file://' not in intermediate_input):
            result = urlparse(intermediate_input)

            # noinspection PyProtectedMember
            inputs = [
                urlunparse(result._replace(path=fragment))
                for fragment in glob.glob(os.path.expanduser(os.path.expandvars(result.path)))
            ]
            
    outputs = []

    for input_filename in inputs:
        outputs.append(output.replace(OUTPUT_WILDCARD, remove_query_if_present(input_filename)))

    overwrite = args.overwrite

    inputs_outputs = check_for_overwriting(inputs, outputs, overwrite)

    if not inputs_outputs:
        log.error("Nothing to do. (Maybe all outputs already existed and --overwrite was not passed?)")

    return inputs_outputs


def prepare_argparser_and_setup(args=None):
    args, parser = get_common_argparser_and_setup(args=args)

    parser.add_argument('--check-connectors', default=False, action='store_true')
    parser.add_argument('--timepoints', default='0-', type=str)
    parser.add_argument('--positions', default='0-', type=str)
    parser.add_argument('--channel', type=int, default=0)
    parser.add_argument('--signature', type=str, default='predict')  # alternatively, take from const
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--output', type=str, help="output", default='{}_segmented.tif')
    parser.add_argument('--output-type',
                        choices=['roi', 'raw', 'rawroi'], default='roi', type=str, help="output_type")

    args = parser.parse_args(args=args)

    return args


# noinspection PyProtectedMember
def main(args=None):
    args = prepare_argparser_and_setup(args)

    if args.check_connectors:
        from .connectors import schema_model_mapping
        for class_ in set(schema_model_mapping.values()):
            working = False
            try:
                class_.check_import()
                working = True
            except ImportError as e:
                log.exception(e)

            log.info("Checked %s, working: %r", class_.__name__, working)
        return

    # establish model

    suggested_connector = suggest_connector(args.model)
    if suggested_connector:
        pass  # set tunable connector to something, if not explicitly set

    if args.model is None:
        raise RuntimeError('Please specify a model/connector argument.')

    mc = ModelConnector(args.model)
    signature = 'predict' if not isinstance(mc, HTTPConnector) else 'predict_png'

    def predict(input_data):
        return mc.call(signature, input_data)

    log.info("Successfully connected %s with following signatures available: %r",
             mc.__class__.__name__,
             mc.get_signatures())

    # prepare inputs

    inputs_outputs = prepare_inputs_outputs(args)

    for input_filename, output_filename in inputs_outputs:
        predict_file_name_with_args(args, input_filename, output_filename, predict)


def predict_file_name_with_args(args, input_filename, output_filename, predict):

    log.info("Opening \"%s\" ...", input_filename)
    ims = ImageStack(input_filename).view(Dimensions.PositionXY, Dimensions.Time, Dimensions.Channel)
    positions = parse_range(args.positions, maximum=ims.size[Dimensions.PositionXY])
    timepoints = parse_range(args.timepoints, maximum=ims.size[Dimensions.Time])
    log.info(
        "Beginning Processing:\n%s\n%s",
        prettify_numpy_array(positions, "Positions : "),
        prettify_numpy_array(timepoints, "Timepoints: ")
    )
    calibration = None
    total_channels = ims.size[Dimensions.Channel]
    channel = args.channel
    overall_output_filename = output_filename

    POSITION_PLACEHOLDER = '{position}'
    if POSITION_PLACEHOLDER not in overall_output_filename and len(positions) > 1:
        if '.tif' in overall_output_filename:
            overall_output_filename = overall_output_filename.replace('.tif', '_pos' + POSITION_PLACEHOLDER + '.tif')
        else:
            overall_output_filename += POSITION_PLACEHOLDER

    position_number_format_str = ("%0" + str(len(str(max(positions)))) + "d")

    output_type = args.output_type

    for n_p, position in tqdm.tqdm(enumerate(positions)):

        output_filename = overall_output_filename.format(position=position_number_format_str % position)
        log.info("Writing position %d to \"%s\" ...", position, output_filename)

        with TiffWriter(output_filename, imagej=True) as tiff_output:
            output_timepoint = -1
            image_buffer = []
            roi_buffer = []

            temporary_input_frame = ims[position, timepoints[0], channel]
            input_frame_pixels = np.prod(temporary_input_frame.shape)

            for n_t, timepoint in tqdm.tqdm(
                    enumerate(timepoints), total=len(timepoints), unit_scale=input_frame_pixels, unit=' pixels'):

                output_timepoint += 1

                with Timed() as time_io_read:

                    input_frame = ims[position, timepoint, channel]
                    calibration = ims.meta[position, timepoint, channel].calibration

                    input_data_for_prediction = input_frame[:, :, None].astype(np.float32)

                with Timed() as time_prediction:

                    prediction = predict(input_data_for_prediction)
                    prediction = np.array(prediction)

                with Timed() as time_io_write:

                    if output_type == 'raw':
                        tiff_output.save(prediction)
                    elif output_type in ('roi', 'rawroi'):

                        detector = ROIDetector()
                        detector.calibration = calibration

                        for contour in detector.get_rois(prediction):
                            kwargs = dict(
                                t=output_timepoint,
                                position=-1
                            ) if total_channels > 1 or output_type == 'rawroi' else dict(
                                t=-1,
                                position=output_timepoint
                            )

                            roi = ImagejRoi.frompoints(contour, c=-1, **kwargs)
                            roi_buffer.append(roi)

                        all_channels = [ims[position, timepoint, inner_channel]
                                        for inner_channel
                                        in range(total_channels)]

                        if output_type == 'rawroi':
                            all_channels = [channel.astype(prediction.dtype) for channel in all_channels]

                            for prediction_channel in range(prediction.shape[-1]):
                                all_channels += [prediction[:, :, prediction_channel]]

                        output_frame = np.stack(all_channels)[np.newaxis]

                        image_buffer.append(output_frame)
                    else:
                        raise RuntimeError('Unsupported output_type')

                log.info(
                    "Predicted position %d/%d timepoint %d/%d, timings: "
                    "Prediction: %.3fs IO read: %.3fs IO write: %.3fs",
                    n_p + 1, len(positions), n_t + 1, len(timepoints),
                    time_prediction, time_io_read, time_io_write)

            roi_buffer = [roi.tobytes() for roi in roi_buffer]

            for frame in tqdm.tqdm(image_buffer):
                tiff_output.save(
                    frame,
                    resolution=(1.0 / calibration, 1.0 / calibration), metadata=dict(unit='um'),
                    ijmetadata=dict(Overlays=roi_buffer)
                )
