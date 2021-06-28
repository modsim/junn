import glob
import logging
import os
from urllib.parse import urlparse, urlunparse

import numpy as np
import pilyso_io.imagestack.readers
import tqdm
from pilyso_io.imagestack import (
    Dimensions,
    ImageStack,
    parse_range,
    prettify_numpy_array,
)
from roifile import ImagejRoi

from ..common.cli import get_common_argparser_and_setup
from ..common.timed import Timed
from .connectors import HTTPConnector, suggest_connector

# from .connectors.model_connector import ModelConnector
from .detectors import ROIDetector

pilyso_io.imagestack.readers = pilyso_io.imagestack.readers

log = logging.getLogger(__name__)


def check_for_overwriting(inputs, outputs, overwrite=False):
    inputs_outputs = []
    for input_filename, output_filename in zip(inputs, outputs):
        if os.path.isfile(output_filename):

            if overwrite:
                log.info("File \"%s\" already exists, overwriting.", output_filename)
            else:
                continue
        inputs_outputs.append(
            (
                input_filename,
                output_filename,
            )
        )
    return inputs_outputs


def remove_query_if_present(input_filename):
    input_filename_to_use = input_filename
    if '?' in input_filename_to_use:
        input_filename_to_use = urlunparse(
            urlparse(input_filename_to_use)._replace(
                params='', query='', fragment='', scheme='', netloc=''
            )
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
                urlunparse(result._replace(path=fragment, fragment=''))
                for fragment in glob.glob(
                    os.path.expanduser(
                        os.path.expandvars(
                            result.path
                            + ('#' + result.fragment if result.fragment else '')
                        )
                    )
                )
            ]

    outputs = []

    for input_filename in inputs:
        outputs.append(
            output.replace(OUTPUT_WILDCARD, remove_query_if_present(input_filename))
        )

    overwrite = args.overwrite

    inputs_outputs = check_for_overwriting(inputs, outputs, overwrite)

    if not inputs_outputs:
        log.error(
            "Nothing to do. "
            "(Maybe all outputs already existed and --overwrite was not passed?)"
        )

    return inputs_outputs


def prepare_argparser_and_setup(args=None):
    args, parser = get_common_argparser_and_setup(args=args)

    parser.add_argument('--check-connectors', default=False, action='store_true')
    parser.add_argument('--timepoints', default='0-', type=str)
    parser.add_argument('--positions', default='0-', type=str)
    parser.add_argument('--channel', type=int, default=0)
    parser.add_argument(
        '--signature', type=str, default='predict'
    )  # alternatively, take from const
    parser.add_argument('--overwrite', default=False, action='store_true')
    parser.add_argument('--timing-precision', default=3, type=int)
    parser.add_argument('--output', type=str, help="output", default='{}_segmented.tif')
    parser.add_argument(
        '--output-type',
        choices=['roi', 'raw', 'rawroi'],
        default='roi',
        type=str,
        help="output_type",
    )

    args = parser.parse_args(args=args)

    return args


# noinspection PyProtectedMember
def main(args=None):
    args = prepare_argparser_and_setup(args)

    Timed.precision = args.timing_precision

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

    # mc = ModelConnector(args.model)
    mc = suggested_connector(args.model)
    signature = 'predict' if not isinstance(mc, HTTPConnector) else 'predict_png'

    def predict(input_data):
        return mc.call(signature, input_data)

    log.info(
        "Successfully connected %s with following signatures available: %r",
        mc.__class__.__name__,
        mc.get_signatures(),
    )

    # prepare inputs

    inputs_outputs = prepare_inputs_outputs(args)

    for input_filename, output_filename in inputs_outputs:
        predict_file_name_with_args(args, input_filename, output_filename, predict)


class OutputWriter:
    calibration = 1.0
    name = ''

    progress = lambda _: _  # noqa: E731

    def __init__(self, name, calibration=1.0):
        self.name = name
        self.calibration = calibration

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._finish()

    def _finish(self):
        pass

    def add_roi(self, roi):
        pass

    def add_frame(self, data):
        pass


class TiffOutputWriter(OutputWriter):
    BACKING_MEMORY = 1
    BACKING_FILE = 2

    backing = BACKING_MEMORY

    def __init__(self, name, calibration=1.0):
        super().__init__(name, calibration=calibration)
        import tifffile  # fail early if it is not installed

        tifffile = tifffile
        self.buffer = []
        self.roi_buffer = []

    def add_roi(self, roi):
        self.roi_buffer.append(roi)

    def add_frame(self, data):
        print(data.shape)
        if self.backing == self.BACKING_MEMORY:
            self.buffer.append(data)
        else:
            raise NotImplementedError

    def _finish(self):
        roi_byte_buffer = [roi.tobytes() for roi in self.roi_buffer]
        from tifffile import TiffWriter

        with TiffWriter(self.name, imagej=True) as tiff_output:
            for frame in self.progress(self.buffer):
                tiff_output.save(
                    frame,
                    resolution=(1.0 / self.calibration, 1.0 / self.calibration),
                    metadata=dict(unit='um', Overlays=roi_byte_buffer),
                )


class ROIZIPWriter(OutputWriter):
    def __init__(self, name, calibration=1.0):
        super().__init__(name, calibration=calibration)
        self.roi_buffer = []

    def add_roi(self, roi):
        self.roi_buffer.append(roi)

    def _finish(self):
        import zipfile

        with zipfile.ZipFile(self.name, 'w', compresslevel=0) as zip_fh:
            for roi in self.roi_buffer:
                with zip_fh.open(roi.name + '.roi', 'w') as roi_fh:
                    roi_fh.write(roi.tobytes())


def suggest_output_writer_from_filename(filename):
    if filename.endswith('.zip'):
        return ROIZIPWriter
    elif filename.endswith('.tif') or filename.endswith('.tiff'):
        return TiffOutputWriter
    else:
        return TiffOutputWriter


def predict_file_name_with_args(args, input_filename, output_filename, predict):

    log.info("Opening \"%s\" ...", input_filename)
    ims = ImageStack(input_filename).view(
        Dimensions.PositionXY, Dimensions.Time, Dimensions.Channel
    )
    positions = parse_range(args.positions, maximum=ims.size[Dimensions.PositionXY])
    timepoints = parse_range(args.timepoints, maximum=ims.size[Dimensions.Time])
    log.info(
        "Beginning Processing:\n%s\n%s",
        prettify_numpy_array(positions, "Positions : "),
        prettify_numpy_array(timepoints, "Timepoints: "),
    )
    calibration = None
    total_channels = ims.size[Dimensions.Channel]
    channel = args.channel
    overall_output_filename = output_filename

    POSITION_PLACEHOLDER = '{position}'
    if POSITION_PLACEHOLDER not in overall_output_filename and len(positions) > 1:
        if '.tif' in overall_output_filename:
            overall_output_filename = overall_output_filename.replace(
                '.tif', '_pos' + POSITION_PLACEHOLDER + '.tif'
            )
        else:
            overall_output_filename += POSITION_PLACEHOLDER

    position_number_format_str = "%0" + str(len(str(max(positions)))) + "d"

    output_type = args.output_type

    for n_p, position in tqdm.tqdm(enumerate(positions)):
        temporary_input_frame = ims[position, timepoints[0], channel]
        input_frame_pixels = np.prod(temporary_input_frame.shape)
        # assumption: the file uses only one calibration
        calibration = ims.meta[position, timepoints[0], channel].calibration

        output_filename = overall_output_filename.format(
            position=position_number_format_str % position
        )
        log.info("Writing position %d to \"%s\" ...", position, output_filename)

        output_writer_class = suggest_output_writer_from_filename(output_filename)
        output_writer_class.progress = tqdm.tqdm

        roi_index = 0

        with output_writer_class(
            name=output_filename, calibration=calibration
        ) as output_writer:
            output_timepoint = -1

            for n_t, timepoint in tqdm.tqdm(
                enumerate(timepoints),
                total=len(timepoints),
                unit_scale=input_frame_pixels,
                unit=' pixels',
            ):

                output_timepoint += 1

                with Timed() as time_io_read:

                    input_frame = ims[position, timepoint, channel]
                    input_data_for_prediction = input_frame[:, :, None].astype(
                        np.float32
                    )

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
                            kwargs = (
                                dict(t=output_timepoint, position=-1)
                                if total_channels > 1 or output_type == 'rawroi'
                                else dict(t=-1, position=output_timepoint)
                            )

                            roi = ImagejRoi.frompoints(
                                contour, c=-1, index=roi_index, **kwargs
                            )
                            output_writer.add_roi(roi)

                            roi_index += 1

                        all_channels = [
                            ims[position, timepoint, inner_channel]
                            for inner_channel in range(total_channels)
                        ]

                        if output_type == 'rawroi':
                            all_channels = [
                                channel.astype(prediction.dtype)
                                for channel in all_channels
                            ]

                            for prediction_channel in range(prediction.shape[-1]):
                                all_channels += [prediction[:, :, prediction_channel]]

                        output_frame = np.stack(all_channels)[np.newaxis]

                        output_writer.add_frame(output_frame)
                    else:
                        raise RuntimeError('Unsupported output_type')

                log.info(
                    "Predicted position %d/%d timepoint %d/%d, timings: "
                    "Prediction: %ss IO read: %ss IO write: %ss",
                    n_p + 1,
                    len(positions),
                    n_t + 1,
                    len(timepoints),
                    time_prediction,
                    time_io_read,
                    time_io_write,
                )
