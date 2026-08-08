"""No API newer than the supported floor may be used outside the compat module.

Checking class names is not enough: Adw.ViewStack has existed since 1.0 but only
learned set_enable_transitions() in 1.7, and calling it on an AppImage carrying
libadwaita 1.5 crashes at startup.  This walks the GObject introspection data,
which records the version every method and property was introduced in.
"""

import ast
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import bigocrpdf

NS = "{http://www.gtk.org/introspection/core/1.0}"
GIR_DIRS = (Path("/usr/share/gir-1.0"), Path("/usr/share/gir-1.0/gir-1.0"))

SRC_ROOT = Path(bigocrpdf.__file__).parent
COMPAT_MODULE = "adw_compat.py"

# Namespace -> (gir file, floor). The floors mirror bigocrpdf._MIN_*_VERSION.
NAMESPACES = {
    "Adw": ("Adw-1.gir", bigocrpdf._MIN_ADW_VERSION),
    "Gtk": ("Gtk-4.0.gir", bigocrpdf._MIN_GTK_VERSION),
}


def _gir_path(filename: str) -> Path | None:
    for directory in GIR_DIRS:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def _members_above_floor(gir: Path, floor: tuple[int, int]) -> dict[str, set[str]]:
    """Map ``ClassName`` -> members introduced after ``floor``."""
    root = ET.parse(gir).getroot()
    result: dict[str, set[str]] = {}
    for node in root.iter():
        if node.tag not in (f"{NS}class", f"{NS}interface"):
            continue
        owner = node.get("name")
        for member in node:
            name, version = member.get("name"), member.get("version")
            if not name or not version:
                continue
            try:
                parsed = tuple(int(part) for part in version.split(".")[:2])
            except ValueError:
                continue
            if parsed > floor and member.tag in (f"{NS}method", f"{NS}property"):
                result.setdefault(owner, set()).add(name.replace("-", "_"))
    return result


def _attribute_receivers(tree: ast.AST) -> dict[str, str]:
    """Map ``self.foo`` / ``foo`` to the widget class it was assigned from.

    Only direct constructions are tracked (``self.stack = Adw.ViewStack()``),
    which is how every widget in this codebase is built.
    """
    receivers: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)):
            continue
        if func.value.id not in NAMESPACES:
            continue
        widget = f"{func.value.id}.{func.attr}"
        for target in node.targets:
            if isinstance(target, ast.Attribute):
                receivers[f"self.{target.attr}"] = widget
            elif isinstance(target, ast.Name):
                receivers[target.id] = widget
    return receivers


def _receiver_name(node: ast.Attribute) -> str | None:
    if isinstance(node.value, ast.Name):
        return node.value.id
    if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
        if node.value.value.id == "self":
            return f"self.{node.value.attr}"
    return None


@pytest.mark.parametrize("namespace", sorted(NAMESPACES))
def test_no_api_newer_than_the_declared_floor(namespace):
    gir_file, floor = NAMESPACES[namespace]
    gir = _gir_path(gir_file)
    if gir is None:
        pytest.skip(f"{gir_file} not installed; introspection data unavailable")

    above_floor = _members_above_floor(gir, floor)

    offenders: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        if path.name == COMPAT_MODULE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        receivers = _attribute_receivers(tree)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            receiver = _receiver_name(node)
            if receiver is None:
                continue
            widget = receivers.get(receiver)
            if widget is None or not widget.startswith(f"{namespace}."):
                continue
            members = above_floor.get(widget.split(".", 1)[1], set())
            if node.attr in members:
                rel = path.relative_to(SRC_ROOT)
                offenders.append(f"{rel}:{node.lineno} {widget}.{node.attr}()")

    floor_text = f"{floor[0]}.{floor[1]}"
    assert not offenders, (
        f"These {namespace} members were introduced after the supported floor "
        f"{floor_text} and must go through utils/{COMPAT_MODULE}, or they will "
        f"crash on an AppImage built against an older stack:\n  " + "\n  ".join(sorted(offenders))
    )


def test_the_scanner_actually_detects_a_known_violation(tmp_path):
    """Guard the guard: a deliberate violation must be reported."""
    gir = _gir_path("Adw-1.gir")
    if gir is None:
        pytest.skip("Adw-1.gir not installed")

    above_floor = _members_above_floor(gir, (1, 5))
    assert "set_enable_transitions" in above_floor.get("ViewStack", set()), (
        "Adw.ViewStack.set_enable_transitions is a 1.7 API and must be seen as "
        "above a 1.5 floor; if this fails the introspection parsing is broken"
    )

    source = "self.stack = Adw.ViewStack()\nself.stack.set_enable_transitions(True)\n"
    tree = ast.parse(source)
    receivers = _attribute_receivers(tree)
    assert receivers == {"self.stack": "Adw.ViewStack"}


def test_compat_module_is_the_only_exemption():
    """The exemption must point at a file that exists."""
    assert (SRC_ROOT / "utils" / COMPAT_MODULE).is_file()


def test_view_stack_transitions_helper_is_used_instead_of_direct_calls():
    offenders = [
        str(path.relative_to(SRC_ROOT))
        for path in SRC_ROOT.rglob("*.py")
        if path.name != COMPAT_MODULE
        and re.search(r"\.set_enable_transitions\s*\(", path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"Use adw_compat.enable_view_stack_transitions() instead: {offenders}"
