"""
Python Version Compatibility Module

Provides utilities for importing modules that may be installed in different
Python versions on the system. This is especially useful when the system
has multiple Python versions (e.g., 3.13 and 3.14) and some packages are
only installed in specific versions.

Key modules that commonly have version compatibility issues:
- rapidocr: OCR library
- openvino: Intel OpenVINO inference engine
- onnxruntime: ONNX Runtime
"""

import importlib
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Known site-packages paths for different Python versions
PYTHON_SITE_PACKAGES_PATTERNS = [
    "/usr/lib/python{version}/site-packages",
    "/usr/local/lib/python{version}/site-packages",
    "/usr/lib64/python{version}/site-packages",
]

# Python versions to search (from newest to oldest)
PYTHON_VERSIONS_TO_SEARCH = ["3.14", "3.13", "3.12", "3.11", "3.10"]

# Modules known to have version compatibility issues
KNOWN_COMPAT_MODULES = [
    "rapidocr",
    "openvino",
    "onnxruntime",
    "cv2",
    "numpy",
]

# Track which paths have been added to avoid duplicates
_added_paths: set[str] = set()


def get_available_python_paths() -> list[Path]:
    """Get all available Python site-packages paths on the system.

    Returns:
        List of existing site-packages directories
    """
    available_paths = []

    for version in PYTHON_VERSIONS_TO_SEARCH:
        for pattern in PYTHON_SITE_PACKAGES_PATTERNS:
            path = Path(pattern.format(version=version))
            if path.exists() and path.is_dir():
                available_paths.append(path)
                logger.debug(f"Found Python site-packages: {path}")

    return available_paths


def find_module_in_paths(module_name: str, paths: list[Path]) -> Path | None:
    """Find a module in the given paths.

    Args:
        module_name: Name of the module to find (e.g., 'rapidocr')
        paths: List of site-packages paths to search

    Returns:
        Path to the module if found, None otherwise
    """
    # Module can be a package (directory) or a single file
    possible_names = [
        module_name,  # Directory package
        f"{module_name}.py",  # Single file module
        module_name.replace(".", "/"),  # Nested module as path
    ]

    for search_path in paths:
        for name in possible_names:
            full_path = search_path / name
            if full_path.exists():
                return search_path
            # Also check for .so files (compiled extensions)
            so_pattern = f"{module_name}*.so"
            so_files = list(search_path.glob(so_pattern))
            if so_files:
                return search_path
            # Check for .pyd files (Windows compiled extensions)
            pyd_pattern = f"{module_name}*.pyd"
            pyd_files = list(search_path.glob(pyd_pattern))
            if pyd_files:
                return search_path

    return None


def add_fallback_paths_to_sys() -> list[str]:
    """Add fallback Python paths to sys.path for module discovery.

    This function adds site-packages from other Python versions to sys.path,
    allowing imports from packages installed in different Python versions.

    Paths are appended (not inserted) so they have lower priority than
    the standard system paths, preventing them from shadowing correct
    packages with incompatible builds from other locations.

    Returns:
        List of paths that were added
    """
    global _added_paths
    added_paths = []
    available_paths = get_available_python_paths()

    for path in available_paths:
        path_str = str(path)
        if path_str not in sys.path and path_str not in _added_paths:
            sys.path.append(path_str)
            _added_paths.add(path_str)
            added_paths.append(path_str)
            logger.info(f"Added fallback Python path: {path_str}")

    return added_paths


def ensure_module_path(module_name: str, try_import: bool = False) -> bool:
    """Ensure the path for a specific module is in sys.path.

    Args:
        module_name: Name of the module to find
        try_import: If True, attempt to import the module to verify it works.
                   Set to False for compiled modules that may not be
                   compatible across Python versions.

    Returns:
        True if path was found and added, False otherwise
    """
    global _added_paths

    # Optionally check if module is already importable
    if try_import:
        try:
            importlib.import_module(module_name)
            return True
        except (ModuleNotFoundError, ImportError):
            pass

    # Search in fallback paths
    available_paths = get_available_python_paths()
    found_path = find_module_in_paths(module_name, available_paths)

    if found_path:
        path_str = str(found_path)
        if path_str not in sys.path:
            sys.path.append(path_str)
            _added_paths.add(path_str)
            logger.info(f"Found '{module_name}' in {path_str}, added to sys.path")
        return True

    return False


def import_with_fallback(module_name: str, package: str | None = None) -> Any:
    """Import a module with fallback to other Python version paths.

    This function first tries a normal import. If that fails, it searches
    for the module in site-packages directories from other Python versions
    and adds the appropriate path before retrying the import.

    Args:
        module_name: Fully qualified module name (e.g., 'rapidocr.RapidOCR')
        package: Package context for relative imports

    Returns:
        The imported module

    Raises:
        ModuleNotFoundError: If module cannot be found in any Python version
    """
    # First, try normal import
    try:
        return importlib.import_module(module_name, package)
    except ModuleNotFoundError:
        logger.warning(
            f"Module '{module_name}' not found in current Python, searching other versions..."
        )

    # Get the base module name (before any dots)
    base_module = module_name.split(".")[0]

    # Search in fallback paths
    available_paths = get_available_python_paths()
    found_path = find_module_in_paths(base_module, available_paths)

    if found_path:
        path_str = str(found_path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            logger.info(f"Found '{base_module}' in {path_str}, added to sys.path")

        # Retry import
        try:
            return importlib.import_module(module_name, package)
        except ModuleNotFoundError:
            logger.error(f"Module '{module_name}' found but failed to import from {path_str}")
            raise

    # If we get here, module was not found anywhere
    logger.error(
        f"Module '{module_name}' not found in any Python version. "
        f"Searched paths: {[str(p) for p in available_paths]}"
    )
    raise ModuleNotFoundError(
        f"No module named '{module_name}'. "
        f"Module not found in current Python ({sys.version_info.major}."
        f"{sys.version_info.minor}) or any fallback versions."
    )


def get_module_from_attribute(module_name: str, *attributes: str) -> tuple[Any, ...]:
    """Import a module and return specific attributes from it.

    This is a convenience function for importing specific items from a module.

    Args:
        module_name: Module to import
        *attributes: Attribute names to extract from the module

    Returns:
        Tuple of requested attributes

    Example:
        RapidOCR, EngineType = get_module_from_attribute(
            'rapidocr', 'RapidOCR', 'EngineType'
        )
    """
    module = import_with_fallback(module_name)
    result = []
    for attr in attributes:
        if hasattr(module, attr):
            result.append(getattr(module, attr))
        else:
            raise AttributeError(f"Module '{module_name}' has no attribute '{attr}'")
    return tuple(result) if len(result) > 1 else result[0]


def setup_python_compatibility() -> None:
    """Setup Python compatibility by adding fallback paths.

    Call this function early in your application startup to ensure
    modules from other Python versions are discoverable.

    This function:
    1. Adds all available Python version site-packages to sys.path
    2. Specifically searches for known compatibility-issue modules (without importing)

    Note: We don't attempt to import modules here because compiled modules
    (like openvino._pyopenvino) may not be compatible across Python versions.
    The paths are added so that when the application actually imports the module,
    it can find the correct version.
    """
    # First, add all fallback paths
    added = add_fallback_paths_to_sys()
    if added:
        logger.info(f"Python compatibility: Added {len(added)} fallback paths")
    else:
        logger.debug("Python compatibility: No additional paths needed")

    # Ensure paths for known problematic modules are added (without importing)
    for module in KNOWN_COMPAT_MODULES:
        ensure_module_path(module, try_import=False)
