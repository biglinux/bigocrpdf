"""Regression tests for image-position ↔ extracted-image correlation.

On a mixed-content page that carries BOTH a full-page scan and a small
decorative image (a logo, a signature stamp), the OCR text of the
full-page scan must land in the scan's display box — not in the logo's tiny
box.  The old matcher sorted extracted images by compressed size and zipped
them with positions by index, which cross-paired the two: the full-page scan's
text got crammed into, e.g., a 75 pt logo box at the bottom of the page.
``match_positions_to_images`` pairs by PDF object number (exact), falling back
to pixel dimensions.
"""

import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pikepdf

from bigocrpdf.services.rapidocr_service.config import ProcessingStats
from bigocrpdf.services.rapidocr_service.pdf_extractor import (
    ImagePosition,
    PdfImageInfo,
    _get_page_xobjects,
    extract_image_positions,
    match_positions_to_images,
    parse_pdfimages_list,
)
from bigocrpdf.services.rapidocr_service.pipeline_mixed_content import (
    MixedContentMixin,
    _index_extracted_images,
)


def _pos(name, w, h, x=0.0, y=0.0):
    return ImagePosition(name=name, page_num=1, x=x, y=y, width=w, height=h)


def _info(idx, w, h, comp, obj=0):
    return PdfImageInfo(
        idx=idx, img_type="image", width=w, height=h, comp_size=comp, object_id=obj
    )


class TestMatchPositionsToImages(unittest.TestCase):
    def test_scan_with_logo_object_id_beats_sort_order(self) -> None:
        """The real cram case: logo first in content stream + larger-comp scan.

        Sort-by-comp_size order would pair the logo box with the scan image.
        Object-number matching must keep each image in its own box.
        """
        positions = [
            _pos("/Xi301", 75, 75, x=-3, y=-65),  # logo display box
            _pos("/wpt1", 595.45, 841.7, x=0, y=0.3),  # full-page scan box
        ]
        infos = [
            _info(0, 125, 125, comp=350, obj=9),  # logo image
            _info(1, 827, 1169, comp=106496, obj=10),  # full-page scan image
        ]
        obj_by_name = {"/Xi301": 9, "/wpt1": 10}
        dims_by_name = {"/Xi301": (125, 125), "/wpt1": (827, 1169)}

        pairs = match_positions_to_images(positions, infos, obj_by_name, dims_by_name)

        self.assertEqual(pairs[0][0].name, "/Xi301")
        self.assertEqual(pairs[0][1].idx, 0)  # logo box <- logo image
        self.assertEqual(pairs[1][0].name, "/wpt1")
        self.assertEqual(pairs[1][1].idx, 1)  # scan box <- scan image (NOT crammed)

    def test_dimension_fallback_when_object_id_unknown(self) -> None:
        positions = [_pos("/A", 75, 75), _pos("/B", 595, 842)]
        infos = [_info(0, 125, 125, comp=350), _info(1, 827, 1169, comp=106496)]
        obj_by_name = {"/A": 0, "/B": 0}  # object ids unknown -> dim fallback
        dims_by_name = {"/A": (125, 125), "/B": (827, 1169)}

        pairs = match_positions_to_images(positions, infos, obj_by_name, dims_by_name)

        self.assertEqual(pairs[0][1].idx, 0)
        self.assertEqual(pairs[1][1].idx, 1)

    def test_object_id_takes_precedence_over_dimensions(self) -> None:
        """When dims collide but object ids differ, object id decides."""
        positions = [_pos("/A", 100, 100), _pos("/B", 100, 100)]
        infos = [_info(0, 50, 50, comp=10, obj=7), _info(1, 50, 50, comp=20, obj=8)]
        obj_by_name = {"/A": 8, "/B": 7}
        dims_by_name = {"/A": (50, 50), "/B": (50, 50)}

        pairs = match_positions_to_images(positions, infos, obj_by_name, dims_by_name)

        self.assertEqual(pairs[0][1].idx, 1)  # /A -> obj 8
        self.assertEqual(pairs[1][1].idx, 0)  # /B -> obj 7

    def test_greedy_single_use_on_colliding_dims(self) -> None:
        """Two identical-dim images, no object ids: each info used at most once."""
        positions = [_pos("/A", 10, 10), _pos("/B", 10, 10)]
        infos = [_info(0, 50, 50, comp=10), _info(1, 50, 50, comp=20)]
        dims_by_name = {"/A": (50, 50), "/B": (50, 50)}

        pairs = match_positions_to_images(positions, infos, {}, dims_by_name)

        matched = [p[1].idx for p in pairs if p[1] is not None]
        self.assertEqual(sorted(matched), [0, 1])  # both used, no double-assignment

    def test_unmatched_position_yields_none(self) -> None:
        positions = [_pos("/A", 10, 10), _pos("/ghost", 10, 10)]
        infos = [_info(0, 50, 50, comp=10, obj=7)]
        obj_by_name = {"/A": 7}
        dims_by_name = {"/A": (50, 50)}

        pairs = match_positions_to_images(positions, infos, obj_by_name, dims_by_name)

        self.assertEqual(pairs[0][1].idx, 0)
        self.assertIsNone(pairs[1][1])


class TestIndexExtractedImages(unittest.TestCase):
    def test_maps_by_global_index_in_filename(self) -> None:
        # idx 1 dropped by the small-PNG filter -> gap must not shift others
        paths = [Path("/x/img-000.png"), Path("/x/img-002.jpg")]
        m = _index_extracted_images(paths)
        self.assertEqual(set(m), {0, 2})
        self.assertEqual(m[2].name, "img-002.jpg")

    def test_accepts_str_paths(self) -> None:
        m = _index_extracted_images(["/x/img-005.ppm"])
        self.assertEqual(set(m), {5})

    def test_ignores_unparseable_names(self) -> None:
        m = _index_extracted_images([Path("/x/thumbnail.png")])
        self.assertEqual(m, {})


class TestParsePdfimagesObjectId(unittest.TestCase):
    def test_parses_object_id_column(self) -> None:
        stdout = (
            "page   num  type   width height color comp bpc  enc interp  "
            "object ID x-ppi y-ppi size ratio\n"
            "----\n"
            "   1     0 image     125   125  gray    1   1  image  no   "
            "       9  0   120   120  350B  18%\n"
            "   1     1 image     827  1169  rgb     3   8  jpeg   no   "
            "      10  0   100   100  104K 3.7%\n"
        )
        fake = SimpleNamespace(returncode=0, stdout=stdout)
        with mock.patch(
            "bigocrpdf.services.rapidocr_service.pdf_extractor.subprocess.run",
            return_value=fake,
        ):
            mapping, masked = parse_pdfimages_list(Path("/does/not/matter.pdf"))
        infos = mapping[1]
        self.assertEqual([i.object_id for i in infos], [9, 10])
        self.assertEqual([i.width for i in infos], [125, 827])
        self.assertEqual(masked, set())


def _two_image_page(path: str) -> None:
    """A page with a SMALL logo drawn first and a LARGE scan drawn second."""
    from PIL import Image
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    def _png(w, h, shade):
        buf = io.BytesIO()
        Image.new("RGB", (w, h), (shade, shade, shade)).save(buf, format="PNG")
        buf.seek(0)
        return ImageReader(buf)

    c = canvas.Canvas(path, pagesize=(612, 792))
    # Small logo first (top-left corner), then full-page scan.
    c.drawImage(_png(100, 120, 200), 20, 700, width=60, height=72)
    c.drawImage(_png(600, 800, 240), 0, 0, width=612, height=792)
    c.showPage()
    c.save()


class _RecordingBackend:
    """Records which extracted file each position was OCR'd with."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(replace_existing_ocr=False, enhance_embedded_images=False)
        self.calls: list[tuple[str, str]] = []

    def _ocr_image_in_page(self, img_path, img_pos, *args, **kwargs):  # noqa: ANN002
        self.calls.append((img_pos.name, Path(img_path).name))
        return []


class TestPlacementWiring(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.pdf_path = os.path.join(self.tmp, "two_images.pdf")
        _two_image_page(self.pdf_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_scan_box_gets_scan_image_not_logo_image(self) -> None:
        positions = extract_image_positions(self.pdf_path)
        page_imgs = positions[1]
        self.assertEqual(len(page_imgs), 2)

        with pikepdf.open(self.pdf_path) as pdf:
            xobjs = _get_page_xobjects(pdf.pages[0])

        # Build a faithful pdfimages map: idx0 = logo file, idx1 = scan file.
        # Identify logo vs scan by actual embedded area (robust to whatever
        # pixel dimensions reportlab stores).  comp_size is set so the OLD
        # sort-by-size matcher would cross-pair (scan first, logo last).
        items = sorted(xobjs.items(), key=lambda kv: kv[1]["width"] * kv[1]["height"])
        (logo_name, logo_d), (scan_name, scan_d) = items[0], items[-1]
        pdfimages_map = {
            1: [
                _info(0, logo_d["width"], logo_d["height"], comp=300, obj=logo_d["obj"]),
                _info(1, scan_d["width"], scan_d["height"], comp=99999, obj=scan_d["obj"]),
            ]
        }
        extracted = [Path(self.tmp, "img-000.png"), Path(self.tmp, "img-001.png")]

        host = _RecordingBackend()
        with pikepdf.open(self.pdf_path) as pdf:
            MixedContentMixin._ocr_image_pages(
                host, pdf, positions, extracted, 2, ProcessingStats(), [], None, None,
                pdfimages_map=pdfimages_map,
            )

        seen = dict(host.calls)
        self.assertEqual(seen[scan_name], "img-001.png")  # scan box <- scan file
        self.assertEqual(seen[logo_name], "img-000.png")  # logo box <- logo file


if __name__ == "__main__":
    unittest.main()
