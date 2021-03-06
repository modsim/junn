from junn.io.tiffmasks import REGION_FOREGROUND, tiff_masks, tiff_peek


def test_tiff_masks(empty_training_data):
    for image, mask in tiff_masks(empty_training_data):
        image = image > 0
        mask = mask == REGION_FOREGROUND
        # assert (image == mask).all()
        # ROI coordinates and pixel values might be off by one, check that


def test_tiff_masks_funny(funny_tiff_file):
    for image, mask in tiff_masks(funny_tiff_file):
        pass


def test_tiff_masks_funny_bytes(funny_tiff_file):
    import numpy as np

    tiff_file_name_bytes = funny_tiff_file.encode('utf8')

    result = tiff_peek(tiff_file_name_bytes)
    assert result.pages == 1
    assert result.h == 512
    assert result.w == 512
    assert result.c == 1
    assert result.dtype == np.uint16

    for image, mask in tiff_masks(tiff_file_name_bytes):
        pass
