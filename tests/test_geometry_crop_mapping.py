import numpy as np

from bigocrpdf.services.rapidocr_service.config import OCRConfig
from bigocrpdf.services.rapidocr_service.ocr_postprocess import apply_ocr_box_offset
from bigocrpdf.services.rapidocr_service.preprocessor import ImagePreprocessor


def test_trim_dark_borders_records_crop_offset():
    config = OCRConfig(
        enable_baseline_dewarp=False, enable_perspective_correction=False, enable_deskew=False
    )
    preprocessor = ImagePreprocessor(config)
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    image[:, :8] = 0
    image[:8, :] = 0

    processed = preprocessor.process(image)

    assert processed.shape[0] < image.shape[0]
    assert processed.shape[1] < image.shape[1]
    assert preprocessor.crop_applied is True
    assert preprocessor.crop_offset_px[0] > 0
    assert preprocessor.crop_offset_px[1] > 0
    assert preprocessor.crop_original_size_px == (100, 100)


def test_apply_ocr_box_offset_shifts_boxes_without_mutating_input():
    raw = {
        "boxes": [
            [[1, 2], [3, 2], [3, 4], [1, 4]],
        ],
        "txts": ["abc"],
        "scores": [0.9],
    }

    shifted = apply_ocr_box_offset(raw, (10, 20))

    assert shifted is not None
    assert shifted["boxes"][0][0] == [11, 22]
    assert raw["boxes"][0][0] == [1, 2]
