from collections import namedtuple

import numpy as np
from roifile import ImagejRoi
from skimage.draw import polygon, polygon_perimeter
from tifffile import TiffFile, TiffWriter

from . import REGION_BACKGROUND, REGION_BORDER, REGION_FOREGROUND

TiffWriter = TiffWriter

TiffInfo = namedtuple('TiffInfo', 'pages, w, h, c, dtype')


def _inner_tiff_peek(tiff):
    first_page = tiff.pages[0]

    imagej_metadata = tiff.imagej_metadata

    if imagej_metadata:

        try:
            count = imagej_metadata['images']
        except KeyError:
            count = 1

        try:
            if imagej_metadata['frames'] < count:
                count = imagej_metadata['frames']
        except KeyError:
            pass
    else:
        count = 1

    return TiffInfo(
        pages=count,
        h=first_page.imagelength,
        w=first_page.imagewidth,
        c=1,
        dtype=first_page.dtype,
    )


def tiff_peek(file_name_or_tiff):
    """
    Fetches some information about a tiff file and returns a TiffInfo named tuple.

    :param file_name_or_tiff:
    :return:
    """

    if isinstance(file_name_or_tiff, TiffFile):
        return _inner_tiff_peek(file_name_or_tiff)
    else:
        file_name = file_name_or_tiff
        if isinstance(file_name_or_tiff, bytes):  # why?!
            file_name = file_name.decode('utf8')
        with TiffFile(file_name) as tiff:
            return _inner_tiff_peek(tiff)


def guess_frame_identifier(all_overlays):
    return (
        't_position'
        if (np.array([overlay.position for overlay in all_overlays]) == 0).all()
        else 'position'
    )


def _get_overlays(all_overlays):
    overlays = {}

    if not isinstance(all_overlays, list):
        all_overlays = [all_overlays]

    all_overlays = [ImagejRoi.frombytes(overlay) for overlay in all_overlays]
    frame_identifier = guess_frame_identifier(all_overlays)

    for overlay in all_overlays:
        frame_number = getattr(overlay, frame_identifier)
        if frame_number not in overlays:
            overlays[frame_number] = []
        overlays[frame_number].append(overlay)

    return overlays


def tiff_to_array(tiff):
    array = (
        tiff.asarray(out='memmap') if tiff.pages[0].is_memmappable else tiff.asarray()
    )
    if array.ndim < 3:
        array = array[np.newaxis, ...]
    return array


def tiff_masks(
    file_name,
    background=REGION_BACKGROUND,
    foreground=REGION_FOREGROUND,
    border=REGION_BORDER,
    skip_empty=False,
):
    """
    Generator, reads a TIFF file with ImageJ ROIs, and yields tuples of image, mask per frame.
    :param file_name:
    :param background:
    :param foreground:
    :param border:
    :param skip_empty:
    :return:
    """

    if isinstance(file_name, bytes):  # why?!
        file_name = file_name.decode('utf8')

    with TiffFile(file_name) as tiff:

        tiff_info = tiff_peek(tiff)
        count = tiff_info.pages

        array = tiff_to_array(tiff)

        if tiff.imagej_metadata:
            overlays = _get_overlays(tiff.imagej_metadata['Overlays'])
        else:
            overlays = {}

        buffer_prototype = np.empty((tiff_info.h, tiff_info.w), dtype=np.uint8)
        buffer_prototype.fill(background)

        for num in range(count):
            buffer = buffer_prototype.copy()

            overlay_num = num + 1

            if overlay_num == 1 and count == 1:
                if 0 in overlays:
                    overlay_num = 0  # weird corner case?
                if 1 in overlays:
                    overlay_num = 1

            if overlay_num not in overlays and skip_empty:
                continue

            if overlay_num in overlays:
                draw_overlays(
                    overlays[overlay_num], buffer, foreground=foreground, border=border
                )

            yield array[num], buffer


def draw_overlays(overlays, buffer, foreground=REGION_FOREGROUND, border=REGION_BORDER):
    foregrounds = np.zeros(buffer.shape, dtype=bool)
    borders = np.zeros(buffer.shape, dtype=bool)

    for overlay in overlays:
        if overlay.name.startswith('TrackMate'):
            continue

        xy = overlay.coordinates()
        xy = xy[:, ::-1]

        if len(xy) < 3:
            continue

        rr, cc = polygon(xy[:, 0], xy[:, 1], shape=buffer.shape)
        foregrounds[rr, cc] = True

        rr, cc = polygon_perimeter(xy[:, 0], xy[:, 1], shape=buffer.shape)
        borders[rr, cc] = True

    buffer[foregrounds] = foreground
    buffer[borders] = border
