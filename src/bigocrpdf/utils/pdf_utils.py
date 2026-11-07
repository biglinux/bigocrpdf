"""PDF page counts, metadata and page rendering through Poppler, and the
sandbox every Poppler tool reading a user's PDF runs in."""

import os
import shutil
import stat
import subprocess
import threading
from collections.abc import Iterable
from contextlib import ExitStack
from functools import cache, lru_cache
from typing import TYPE_CHECKING, Any

from bigocrpdf.utils.logger import logger

if TYPE_CHECKING:
    from PIL import Image, JpegImagePlugin

# What a confined Poppler tool sees besides the files it is given: the system
# read-only, fonts, a private /tmp, and no network, IPC, other processes or
# home directory beyond the user's fonts (``_user_font_binds``).
_SANDBOX_PROFILE = (
    "--unshare-all",
    "--die-with-parent",
    "--new-session",
    "--clearenv",
    "--setenv",
    "LANG",
    "C.UTF-8",
    "--proc",
    "/proc",
    "--dev",
    "/dev",
    "--tmpfs",
    "/tmp",
    "--ro-bind-try",
    "/etc/fonts",
    "/etc/fonts",
    "--ro-bind-try",
    "/var/cache/fontconfig",
    "/var/cache/fontconfig",
)


# The host's top-level program and library directories, often merged-/usr links.
_TOP_LEVEL_ROOTS = ("/bin", "/sbin", "/lib", "/lib32", "/lib64")
# Where the system's programs and libraries live: bound read-only, so a
# tool found there needs no bind of its own.
_SYSTEM_ROOTS = ("/usr", "/nix/store", *_TOP_LEVEL_ROOTS)


@cache
def _system_binds() -> tuple[str, ...]:
    """The read-only system: /usr, the Nix store and the loader cache when
    they exist, and the host's top-level /bin, /sbin and /lib* as they are,
    merged-/usr links kept as links."""
    binds = ["--ro-bind-try", "/usr", "/usr", "--ro-bind-try", "/nix/store", "/nix/store"]
    binds += ["--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache"]
    for path in _TOP_LEVEL_ROOTS:
        if os.path.islink(path):
            binds += ["--symlink", os.readlink(path), path]
        elif os.path.isdir(path):
            binds += ["--ro-bind", path, path]
    return tuple(binds)


@cache
def _user_font_binds() -> tuple[str, ...]:
    """HOME and the XDG base directories at the host's paths, so fontconfig
    resolves the user's fonts, settings and font cache as outside, with only
    those bound in, read-only: a PDF may name a font it does not embed."""
    home = os.path.expanduser("~")
    binds = ["--setenv", "HOME", home]
    base = {}
    for name, default in (
        ("XDG_DATA_HOME", ".local/share"),
        ("XDG_CONFIG_HOME", ".config"),
        ("XDG_CACHE_HOME", ".cache"),
    ):
        value = os.environ.get(name, "")
        base[name] = value if os.path.isabs(value) else os.path.join(home, default)
        binds += ["--setenv", name, base[name]]
    fonts = (
        os.path.join(base["XDG_DATA_HOME"], "fonts"),
        os.path.join(home, ".fonts"),
        os.path.join(base["XDG_CONFIG_HOME"], "fontconfig"),
        os.path.join(base["XDG_CACHE_HOME"], "fontconfig"),
    )
    # Where each would be as a plain directory: a link to anywhere else in the
    # home directory, or to the home directory or above it, is left out.
    plain = [os.path.join(os.path.realpath(os.path.dirname(f)), os.path.basename(f)) for f in fonts]
    real_home = os.path.realpath(home)
    for path in fonts:
        if not os.path.isdir(path):
            continue
        real = os.path.realpath(path)
        in_home = os.path.commonpath([real, real_home]) in (real, real_home)
        if in_home and not any(os.path.commonpath([real, p]) == p for p in plain):
            logger.warning(f"Not binding {path} into the PDF sandbox: it is {real}")
            continue
        binds += ["--ro-bind", real, path]
    return tuple(binds)


def _sandboxed(
    bwrap: str,
    tool: str,
    args: list[str],
    read: Iterable[str | os.PathLike[str]],
    write: Iterable[str | os.PathLike[str]],
) -> list[str]:
    """``tool`` (a resolved path) with ``args`` in bwrap, seeing ``read``
    read-only and ``write`` read-write. Read binds come last, so a file read
    from inside a writable directory stays read-only."""
    command = [bwrap, *_SANDBOX_PROFILE, *_system_binds(), *_user_font_binds()]
    if not tool.startswith(tuple(root + "/" for root in _SYSTEM_ROOTS)):
        command += ["--ro-bind", tool, tool]
    for flag, paths in (("--bind", write), ("--ro-bind", read)):
        for path in paths:
            name = os.path.abspath(path)
            command += [flag, os.path.realpath(name), name]
    return [*command, "--", tool, *args]


# The probe's result, guarded by its lock so concurrent first callers probe once.
_SANDBOX_LOCK = threading.Lock()
_sandbox: tuple[str | None, str] | None = None


def sandbox_state() -> tuple[str | None, str]:
    """``(bwrap, "bwrap")`` when Poppler tools can run confined, else
    ``(None, "none (<reason>)")``; probed once per process, by running
    ``pdfinfo -v`` the way every tool runs."""
    global _sandbox
    with _SANDBOX_LOCK:
        if _sandbox is None:
            _sandbox = _probe_sandbox()
        return _sandbox


def _probe_sandbox() -> tuple[str | None, str]:
    if os.environ.get("FLATPAK_ID"):
        reason = "flatpak"
    elif (bwrap := shutil.which("bwrap")) is None:
        reason = "bwrap not installed"
    elif (pdfinfo := shutil.which("pdfinfo")) is None:
        reason = "no pdfinfo to probe with"
    else:
        probe = _sandboxed(bwrap, os.path.realpath(pdfinfo), ["-v"], (), ())
        try:
            result = subprocess.run(probe, capture_output=True, text=True, check=False, timeout=10)
        except (OSError, subprocess.SubprocessError) as error:
            reason = f"bwrap failed: {error}"
        else:
            if result.returncode == 0:
                return bwrap, "bwrap"
            reason = f"bwrap failed: {result.stderr.strip()[:200]}"
    logger.warning(f"PDF tools in this process run without a sandbox: {reason}")
    return None, f"none ({reason})"


def confined(
    argv: list[str],
    *,
    read: Iterable[str | os.PathLike[str]] = (),
    write: Iterable[str | os.PathLike[str]] = (),
) -> list[str]:
    """``argv`` of a Poppler tool, run in a bwrap sandbox that sees only the
    files in ``read`` (read-only) and the directories in ``write``.

    Paths are bound at their absolute names, which the tool must be given;
    ``argv[0]`` is resolved here, so the sandbox needs no PATH. Without a
    usable bwrap the command runs as given. A tool that fails inside the
    sandbox has failed: callers never retry it unconfined.
    """
    bwrap, _state = sandbox_state()
    if bwrap is None:
        return argv
    tool = shutil.which(argv[0])
    if tool is None:
        return argv
    return _sandboxed(bwrap, os.path.realpath(tool), argv[1:], read, write)


def is_regular_file(path: str | os.PathLike[str]) -> bool:
    """Whether ``path`` is a regular file itself, not a link to one: what a
    confined tool wrote is read only then, so a tool that plants a link
    cannot hand the app another file of the user's."""
    try:
        return stat.S_ISREG(os.lstat(path).st_mode)
    except OSError:
        return False


def get_pdf_page_count(file_path: str, use_cache: bool = True) -> int:
    """Get the number of pages in a PDF file using pdfinfo.

    Args:
        file_path: Path to the PDF file
        use_cache: Whether to use cached values (default: True)

    Returns:
        Number of pages, or 0 if unable to determine
    """
    if not file_path or not os.path.exists(file_path):
        return 0

    if not use_cache:
        return _get_page_count_uncached(file_path)
    try:
        info = os.stat(file_path)
    except OSError:
        return 0
    return _get_page_count_cached(file_path, info.st_mtime_ns, info.st_size)


@lru_cache(maxsize=256)
def _get_page_count_cached(file_path: str, _mtime_ns: int, _size: int) -> int:
    return _get_page_count_uncached(file_path)


def _get_page_count_uncached(file_path: str) -> int:
    """Get page count without caching.

    Args:
        file_path: Path to the PDF file

    Returns:
        Number of pages, or 0 if unable to determine
    """
    try:
        source = os.path.abspath(file_path)
        result = subprocess.run(
            confined(["pdfinfo", source], read=[source]),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=10,  # Prevent hanging on corrupted files
        )

        for line in result.stdout.split("\n"):
            if line.startswith("Pages:"):
                return int(line.split(":")[1].strip())

    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout getting page count for {os.path.basename(file_path)}")
    except ValueError as e:
        logger.warning(f"Invalid page count value in {os.path.basename(file_path)}: {e}")
    except Exception as e:
        logger.error(f"Error getting page count for {os.path.basename(file_path)}: {e}")

    return 0


def get_pdf_thumbnail(file_path: str, max_size: int = 200) -> bytes | None:
    """Get a thumbnail of the first page of a PDF.

    Args:
        file_path: Path to the PDF file
        max_size: Maximum width/height in pixels (default: 200)

    Returns:
        PNG image bytes or None if extraction failed
    """
    if not file_path or not os.path.exists(file_path):
        return None

    return render_pdf_page_to_png(file_path, page_num=0, size=max_size)


def get_pdf_info(file_path: str) -> dict[str, Any]:
    """Get metadata about a PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        Dictionary with PDF info (pages, title, author, file_size, etc.)
    """
    info: dict[str, Any] = {
        "pages": 0,
        "title": "",
        "author": "",
        "file_size": 0,
        "file_size_mb": 0.0,
    }

    if not file_path or not os.path.exists(file_path):
        return info

    try:
        size = os.path.getsize(file_path)
        info["file_size"] = size
        info["file_size_mb"] = round(size / (1024 * 1024), 2)
    except OSError:
        pass

    # Get page count and metadata from one pdfinfo snapshot.
    try:
        source = os.path.abspath(file_path)
        result = subprocess.run(
            confined(["pdfinfo", source], read=[source]),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=10,
        )

        for line in result.stdout.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                value = value.strip()

                if key == "pages":
                    try:
                        info["pages"] = int(value)
                    except ValueError:
                        logger.debug("Invalid PDF page count: %s", value)
                elif key == "title" and value:
                    info["title"] = value
                elif key == "author" and value:
                    info["author"] = value

    except (OSError, subprocess.SubprocessError) as e:
        logger.debug(f"Could not get PDF metadata: {e}")

    return info


def render_pdf_page_to_png(
    pdf_path: str,
    page_num: int,
    size: int = 200,
) -> bytes | None:
    """PNG bytes of one page (0-indexed), its longer side ``size`` pixels,
    rendered by a confined pdftoppm.

    Returns:
        PNG image bytes, or None if rendering failed
    """
    source = os.path.abspath(pdf_path)
    page = str(page_num + 1)
    command = ["pdftoppm", "-png", "-scale-to", str(size), "-f", page, "-l", page, "-singlefile"]
    try:
        result = subprocess.run(
            confined([*command, source], read=[source]),
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.error(f"Error rendering PDF page: {e}")
        return None
    if result.returncode != 0 or not result.stdout:
        logger.error(f"pdftoppm could not render page {page}: {result.stderr[:200]!r}")
        return None
    return result.stdout


# Pillow decoders allowed on bytes from outside: the formats screenshots,
# pdfimages, pdftoppm and our own outputs come in. Pillow picks a plugin from
# the content, whatever the file is called, and some (EPS, through Ghostscript)
# run external code.
_SAFE_IMAGE_FORMATS = frozenset(
    ("AVIF", "BMP", "GIF", "JPEG", "JPEG2000", "PNG", "PPM", "TIFF", "WEBP")
)


def safe_image_formats() -> tuple[str, ...]:
    """The ``_SAFE_IMAGE_FORMATS`` this Pillow build can decode."""
    from PIL import Image

    Image.init()
    return tuple(name for name in sorted(_SAFE_IMAGE_FORMATS) if name in Image.OPEN)


def open_image(source: Any) -> "Image.Image":
    """``PIL.Image.open`` limited to ``_SAFE_IMAGE_FORMATS``: any other format
    raises ``UnidentifiedImageError``."""
    from PIL import Image

    return Image.open(source, formats=safe_image_formats())


def is_jpeg(source: Any) -> bool:
    """Whether a path's file, or a binary stream from where it stands,
    starts with a JPEG start-of-image marker."""
    if hasattr(source, "read"):
        start = source.tell()
        head = source.read(3)
        source.seek(start)
        return head == b"\xff\xd8\xff"
    with open(source, "rb") as f:
        return f.read(3) == b"\xff\xd8\xff"


def open_image_header(source: Any) -> "Image.Image":
    """An image opened for its size, format, EXIF and JPEG tables, never
    decoded. A JPEG skips Pillow's decompression-bomb check, which guards
    decoding, so a phone photo of any size reads; anything else opens as
    ``open_image``."""
    from PIL import JpegImagePlugin

    if is_jpeg(source):
        return JpegImagePlugin.JpegImageFile(source)
    return open_image(source)


# The most pixels a JPEG may store to be decoded at a reduced scale: 200 MP
# phone photos and some margin. A baseline file is decoded one row of blocks
# at a time, so its memory follows the reduced size, but every block is still
# entropy-decoded, so its decode time follows the stored size.
MAX_JPEG_STORED_PIXELS = 256_000_000


def open_jpeg_within(
    source: Any, full_pixels: float, reduced_pixels: float
) -> "tuple[JpegImagePlugin.JpegImageFile, int]":
    """A JPEG opened to decode whole when it holds at most ``full_pixels``,
    else at the largest of libjpeg's 1/2, 1/4 and 1/8 scales that holds at
    most ``reduced_pixels``, and that scale's divisor (1 when whole);
    ValueError when no scale fits, when it stores over
    ``MAX_JPEG_STORED_PIXELS``, or when a progressive file would need a
    reduced scale: libjpeg keeps all of a progressive file's coefficients.

    The scale bounds the decode, so Pillow's decompression-bomb check, which
    ``Image.open`` makes on the stored size, is not needed here; this opens
    JPEG only and raises SyntaxError on anything else.
    """
    from PIL import JpegImagePlugin

    image = JpegImagePlugin.JpegImageFile(source)
    width, height = image.size
    if width * height <= full_pixels:
        return image, 1
    if width * height > MAX_JPEG_STORED_PIXELS or image.info.get("progressive"):
        kind = "progressive JPEG" if image.info.get("progressive") else "JPEG"
        image.close()
        raise ValueError(f"{kind} of {width}x{height} is too large to decode reduced")
    for scale in (2, 4, 8):
        # libjpeg rounds a scaled size up; draft() picks the largest scale
        # the requested size allows, so it is asked for the rounded-down one.
        if -(-width // scale) * -(-height // scale) <= reduced_pixels:
            image.draft(None, (max(width // scale, 1), max(height // scale, 1)))
            if image.size[0] * image.size[1] <= reduced_pixels:
                return image, scale
            break
    image.close()
    raise ValueError(
        f"JPEG of {width}x{height} does not decode within "
        f"{reduced_pixels / 1_000_000:g} MP even at 1/8 scale"
    )


# Image extensions recognized as valid inputs.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".avif")


def is_image_file(file_path: str) -> bool:
    """Check if a file path has a supported image extension.

    Args:
        file_path: Path to check

    Returns:
        True if the file is a supported image type
    """
    return file_path.lower().endswith(IMAGE_EXTENSIONS)


def images_to_pdf(image_paths: list[str], output_path: str | None = None) -> str:
    """Convert one or more images to a single PDF file.

    Uses Pillow to convert images, applying EXIF rotation and converting
    RGBA/LA modes to RGB for PDF compatibility.

    Args:
        image_paths: List of image file paths to convert
        output_path: Optional output path. If None, a temp file in /tmp is created.

    Returns:
        Path to the created PDF file

    Raises:
        ValueError: If no valid images provided
        RuntimeError: If conversion fails
    """
    from PIL import Image, ImageOps

    if not image_paths:
        raise ValueError("No image paths provided")

    with ExitStack() as resources:
        images: list[Image.Image] = []
        for path in image_paths:
            try:
                source = resources.enter_context(open_image(path))
                image = ImageOps.exif_transpose(source)
                if image is not source:
                    resources.callback(image.close)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                    resources.callback(image.close)
                images.append(image)
            except Exception as e:
                logger.error(f"Failed to open image {path}: {e}")

        if not images:
            raise ValueError("No valid images could be opened")

        owns_output = output_path is None
        if owns_output:
            from bigocrpdf.utils.temp_manager import mkstemp

            base = os.path.splitext(os.path.basename(image_paths[0]))[0]
            if len(image_paths) > 1:
                base = f"{base}_+{len(image_paths) - 1}"
            fd, output_path = mkstemp(prefix=f"bigocr_{base}_", suffix=".pdf")
            os.close(fd)

        try:
            first_img = images[0]
            rest = images[1:]
            first_img.save(output_path, format="PDF", save_all=True, append_images=rest)
        except Exception as e:
            if owns_output:
                from bigocrpdf.utils.temp_manager import remove_file

                remove_file(output_path)
            raise RuntimeError(f"Failed to create PDF from images: {e}") from e

        logger.info(f"Created PDF from {len(images)} image(s): {output_path}")
        return output_path


# Map the user-facing page-layout setting to a PDF catalog ``/PageLayout``
# name (ISO 32000 §7.7.2). ``"default"`` omits the key so the viewer chooses.
PAGE_LAYOUT_NAMES: dict[str, str] = {
    "single": "/SinglePage",
    "continuous": "/OneColumn",
    "two_page": "/TwoColumnLeft",
}


def set_root_page_layout(pdf: Any, page_layout: str) -> bool:
    """Set the document catalog ``/PageLayout`` from the page-layout setting.

    Controls how viewers (notably mobile readers) arrange pages on open:
    one page at a time, continuous vertical scroll, or two-up. ``"default"``
    (or any unknown value) leaves the catalog untouched so the viewer decides.

    ``pdf`` is a ``pikepdf.Pdf``. Returns True when a name was written.
    """
    import pikepdf

    name = PAGE_LAYOUT_NAMES.get(page_layout)
    if name is None:
        return False
    pdf.Root.PageLayout = pikepdf.Name(name)
    return True
