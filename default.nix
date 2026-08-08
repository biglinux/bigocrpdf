{
  lib,
  python3Packages,
  gtk4,
  libadwaita,
  pkg-config,
  wrapGAppsHook4,
  gobject-introspection,
  gettext,
  poppler_utils,
  ghostscript,
  jbig2enc ? null,
}:

assert lib.versionAtLeast python3Packages.python.version "3.12";
assert lib.versionAtLeast gtk4.version "4.22";
assert lib.versionAtLeast libadwaita.version "1.8";
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
    gettext
    pkg-config
    wrapGAppsHook4
    gobject-introspection
  ];

  buildInputs = [
    gtk4
    libadwaita
    poppler_utils
    ghostscript
  ] ++ (if jbig2enc != null then [ jbig2enc ] else []);

  postInstall = ''
    bash $src/tools/stage-data.sh "$out"
  '';

  meta = {
    description = "OCR toolkit for Linux — searchable PDFs, image OCR, PDF editor";
    homepage = "https://github.com/biglinux/bigocrpdf";
    license = lib.licenses.gpl3Plus;
    mainProgram = "bigocrpdf";
    platforms = lib.platforms.linux;
  };
}
