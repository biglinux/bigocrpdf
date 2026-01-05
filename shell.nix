# Shell environment for development
# Usage: nix-shell or nix develop
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Python environment
    python3
    python3Packages.pygobject3
    python3Packages.pycairo
    python3Packages.ocrmypdf
    
    # GTK4 and Adwaita
    gtk4
    libadwaita
    gobject-introspection
    
    # OCR dependencies
    tesseract
    ocrmypdf
    poppler_utils
    
    # Development tools
    ruff
    pkg-config
  ];

  shellHook = ''
    echo "BigOcrPdf development environment loaded"
    echo "Run with: python -m bigocrpdf"
    export PYTHONPATH="$PWD/src:$PYTHONPATH"
  '';
}
