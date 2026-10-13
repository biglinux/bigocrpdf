# Shell environment for development
# Usage: nix-shell or nix develop
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Python environment
    python3
    python3Packages.pygobject3
    python3Packages.pycairo
    python3Packages.pikepdf
    python3Packages.reportlab
    python3Packages.opencv4
    python3Packages.pillow
    python3Packages.numpy
    python3Packages.scipy
    python3Packages.odfpy
    python3Packages.pytest

    # GTK4 and Adwaita
    gtk4
    libadwaita
    gobject-introspection

    # Runtime tools; ppocr-native has no nixpkgs package and must be on PATH
    poppler
    poppler_utils
    gettext
    noto-fonts

    # Development tools
    ruff
    pkg-config
  ];

  shellHook = ''
    echo "BigOcrPdf development environment loaded"
    echo "Run with: python -m bigocrpdf"
    export PYTHONPATH="$PWD/src:$PYTHONPATH"
    export BIGOCRPDF_TEXT_FONT="${pkgs.noto-fonts}/share/fonts/noto/NotoSans.ttf"
  '';
}
