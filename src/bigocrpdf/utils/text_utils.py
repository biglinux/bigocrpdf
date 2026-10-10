"""Reading OCR text from sidecar .txt files."""

from bigocrpdf.utils.durable_writes import read_regular_file_bytes
from bigocrpdf.utils.logger import logger


def read_text_from_sidecar(sidecar_path: str) -> str | None:
    """Read text from a sidecar .txt file.

    Args:
        sidecar_path: Path to the sidecar text file

    Returns:
        Text content, or None if file doesn't exist or can't be read
    """
    if not sidecar_path:
        return None

    try:
        content = read_regular_file_bytes(sidecar_path)
    except FileNotFoundError:
        return None
    except OSError as e:
        logger.error(f"Error reading sidecar file: {e}")
        return None

    try:
        text = content.decode("utf-8")
        encoding = "UTF-8"
    except UnicodeDecodeError:
        text = content.decode("latin-1")
        encoding = "Latin-1"

    if text.strip():
        logger.info("Read %d characters from %s sidecar", len(text), encoding)
        return text

    return None
