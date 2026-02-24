{
  python3Packages,
  gtk4,
  libadwaita,
  pkg-config,
  wrapGAppsHook4,
  gobject-introspection,
  poppler_utils,
  ghostscript,
  fribidi,
  jbig2enc ? null,
}:

python3Packages.buildPythonApplication {
  pname = "bigocrpdf";
  version = "3.0.0";

  src = ./.;

  pyproject = true;

  build-system = with python3Packages; [ setuptools wheel ];

  dependencies = with python3Packages; [
    pygobject3
    pycairo
    rapidocr
    pikepdf
    reportlab
    opencv4
    pillow
    numpy
    scipy
    odfpy
  ];

  nativeBuildInputs = [
    pkg-config
    wrapGAppsHook4
    gobject-introspection
  ];

  buildInputs = [
    gtk4
    libadwaita
    poppler_utils
    ghostscript
    fribidi
  ] ++ (if jbig2enc != null then [ jbig2enc ] else []);

  postInstall = ''
    # Install desktop files
    mkdir -p $out/share/applications
    cp $src/usr/share/applications/*.desktop $out/share/applications/ || true

    # Install icons
    mkdir -p $out/share/icons
    cp -r $src/usr/share/icons/* $out/share/icons/ || true

    # Install service menus
    mkdir -p $out/share/kio/servicemenus
    cp $src/usr/share/kio/servicemenus/*.desktop $out/share/kio/servicemenus/ || true

    # Install locale files
    mkdir -p $out/share/locale
    cp -r $src/usr/share/locale/* $out/share/locale/ || true

    # Install bin wrappers
    mkdir -p $out/bin
    cp $src/usr/bin/* $out/bin/ || true
  '';

  meta = {
    description = "OCR toolkit for Linux — searchable PDFs, image OCR, PDF editor";
    homepage = "https://github.com/biglinux/bigocrpdf";
    license = "GPL-3.0-or-later";
    mainProgram = "bigocrpdf";
  };
}
