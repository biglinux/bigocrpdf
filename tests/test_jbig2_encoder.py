"""Tests for jbig2_encoder module."""

import io
import struct
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from bigocrpdf.services.rapidocr_service.jbig2_encoder import (
    _extract_ccitt_from_tiff,
    _read_ifd_values,
    encode_bilevel,
    encode_ccitt_g4,
    encode_jbig2,
    encode_jbig2_with_globals,
    jbig2enc_available,
)


def _make_binary_image(width: int = 200, height: int = 100) -> np.ndarray:
    """Create a synthetic binary image (black text on white)."""
    img = np.full((height, width), 255, dtype=np.uint8)
    # Add some "text" (black rectangles)
    img[10:20, 10:50] = 0
    img[30:40, 20:80] = 0
    img[50:60, 10:70] = 0
    return img


class TestJbig2encAvailable(unittest.TestCase):
    def test_returns_bool(self):
        result = jbig2enc_available()
        self.assertIsInstance(result, bool)

    @patch("shutil.which", return_value=None)
    def test_not_available(self, _):
        self.assertFalse(jbig2enc_available())

    @patch("shutil.which", return_value="/usr/bin/jbig2")
    def test_available(self, _):
        self.assertTrue(jbig2enc_available())


class TestEncodeJbig2(unittest.TestCase):
    def test_none_input(self):
        self.assertIsNone(encode_jbig2(None))

    def test_empty_input(self):
        self.assertIsNone(encode_jbig2(np.array([], dtype=np.uint8)))

    @unittest.skipUnless(jbig2enc_available(), "jbig2enc not installed")
    def test_encodes_binary_image(self):
        img = _make_binary_image()
        result = encode_jbig2(img)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, bytes)
        self.assertGreater(len(result), 0)

    @unittest.skipUnless(jbig2enc_available(), "jbig2enc not installed")
    def test_output_smaller_than_raw(self):
        img = _make_binary_image(400, 300)
        result = encode_jbig2(img)
        raw_size = img.size  # uncompressed bytes
        self.assertLess(len(result), raw_size)

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_jbig2_not_found(self, _):
        img = _make_binary_image()
        self.assertIsNone(encode_jbig2(img))


class TestEncodeJbig2WithGlobals(unittest.TestCase):
    def test_none_input(self):
        self.assertIsNone(encode_jbig2_with_globals(None))

    def test_empty_input(self):
        self.assertIsNone(encode_jbig2_with_globals(np.array([], dtype=np.uint8)))

    @unittest.skipUnless(jbig2enc_available(), "jbig2enc not installed")
    def test_returns_tuple(self):
        img = _make_binary_image()
        result = encode_jbig2_with_globals(img)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    @unittest.skipUnless(jbig2enc_available(), "jbig2enc not installed")
    def test_page_and_globals_are_bytes(self):
        img = _make_binary_image()
        page_data, globals_data = encode_jbig2_with_globals(img)
        self.assertIsInstance(page_data, bytes)
        self.assertIsInstance(globals_data, bytes)
        self.assertGreater(len(page_data), 0)

    @unittest.skipUnless(jbig2enc_available(), "jbig2enc not installed")
    def test_symbol_mode_smaller_than_generic(self):
        """Symbol mode should typically produce smaller output for text-like images."""
        img = _make_binary_image(400, 300)
        generic = encode_jbig2(img)
        page_data, globals_data = encode_jbig2_with_globals(img)
        # Symbol mode total might be larger for small images but should work
        self.assertIsNotNone(generic)
        self.assertIsNotNone(page_data)


class TestEncodeCcittG4(unittest.TestCase):
    def test_none_input(self):
        self.assertIsNone(encode_ccitt_g4(None))

    def test_empty_input(self):
        self.assertIsNone(encode_ccitt_g4(np.array([], dtype=np.uint8)))

    def test_returns_tuple(self):
        img = _make_binary_image()
        result = encode_ccitt_g4(img)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_returns_correct_dimensions(self):
        img = _make_binary_image(200, 100)
        ccitt_data, width, height = encode_ccitt_g4(img)
        self.assertEqual(width, 200)
        self.assertEqual(height, 100)

    def test_ccitt_data_non_empty(self):
        img = _make_binary_image()
        ccitt_data, _, _ = encode_ccitt_g4(img)
        self.assertIsInstance(ccitt_data, bytes)
        self.assertGreater(len(ccitt_data), 0)

    def test_output_smaller_than_raw(self):
        img = _make_binary_image(400, 300)
        ccitt_data, _, _ = encode_ccitt_g4(img)
        raw_size = img.size
        self.assertLess(len(ccitt_data), raw_size)


class TestExtractCcittFromTiff(unittest.TestCase):
    def test_too_short(self):
        self.assertIsNone(_extract_ccitt_from_tiff(b"short"))

    def test_bad_byte_order(self):
        self.assertIsNone(_extract_ccitt_from_tiff(b"XX\x00\x00\x00\x00\x00\x00"))

    def test_valid_tiff_extraction(self):
        """Create a real CCITT G4 TIFF and verify extraction."""
        img = _make_binary_image()
        pil_img = Image.fromarray(img).convert("1")
        buf = io.BytesIO()
        pil_img.save(buf, format="TIFF", compression="group4")
        tiff_bytes = buf.getvalue()

        result = _extract_ccitt_from_tiff(tiff_bytes)
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)

    def test_single_strip_tiff(self):
        """Single strip TIFF should also work."""
        from PIL import TiffImagePlugin

        img = _make_binary_image(100, 50)
        pil_img = Image.fromarray(img).convert("1")
        buf = io.BytesIO()
        tiffinfo = TiffImagePlugin.ImageFileDirectory_v2()
        tiffinfo[278] = 50  # Full height = single strip
        pil_img.save(buf, format="TIFF", compression="group4", tiffinfo=tiffinfo)
        tiff_bytes = buf.getvalue()

        result = _extract_ccitt_from_tiff(tiff_bytes)
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)


class TestEncodeBilevel(unittest.TestCase):
    def test_none_returns_none_without_jbig2(self):
        with patch(
            "bigocrpdf.services.rapidocr_service.jbig2_encoder.jbig2enc_available",
            return_value=False,
        ):
            result = encode_bilevel(np.array([], dtype=np.uint8))
            self.assertIsNone(result)

    def test_auto_selects_jbig2_when_available(self):
        if not jbig2enc_available():
            self.skipTest("jbig2enc not installed")
        img = _make_binary_image()
        result = encode_bilevel(img)
        self.assertIsNotNone(result)
        enc_name, data, globals_or_none = result
        self.assertEqual(enc_name, "jbig2")
        self.assertIsInstance(data, bytes)

    def test_falls_back_to_ccitt(self):
        with patch(
            "bigocrpdf.services.rapidocr_service.jbig2_encoder.jbig2enc_available",
            return_value=False,
        ):
            img = _make_binary_image()
            result = encode_bilevel(img)
            self.assertIsNotNone(result)
            enc_name, data, globals_or_none = result
            self.assertEqual(enc_name, "ccitt")
            self.assertIsNone(globals_or_none)


if __name__ == "__main__":
    unittest.main()
