{
  lib,
  python3Packages,
  gtk4,
  libadwaita,
  pkg-config,
  wrapGAppsHook4,
  gobject-introspection,
  gettext,
  poppler,
  poppler_utils,
  noto-fonts,
  wqy_microhei,
  jbig2enc ? null,
  # The OCR engine, built from github.com/biglinux/ppocr-native; nixpkgs has no
  # package for it, so the caller passes one. Without it the application starts
  # and reports the engine missing.
  ppocr-native ? null,
}:

assert lib.versionAtLeast python3Packages.python.version "3.12";
assert lib.versionAtLeast gtk4.version "4.14";
assert lib.versionAtLeast libadwaita.version "1.5";
python3Packages.buildPythonApplication {
  pname = "bigocrpdf";
  version = "3.0.0";

  src = ./.;

  pyproject = true;

  build-system = with python3Packages; [ setuptools wheel ];

  dependencies = with python3Packages; [
    pygobject3
    pycairo
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
    poppler
  ];

  # The application runs the Poppler tools and jbig2 from PATH. nixpkgs
  # installs Noto Sans as one variable font with the axis tag removed from its
  # name.
  makeWrapperArgs = [
    "--prefix" "PATH" ":" (lib.makeBinPath ([ poppler_utils ] ++ lib.optional (jbig2enc != null) jbig2enc))
    "--set-default" "BIGOCRPDF_TEXT_FONT" "${noto-fonts}/share/fonts/noto/NotoSans.ttf"
    "--set-default" "BIGOCRPDF_CJK_FONT" "${wqy_microhei}/share/fonts/truetype/wqy-microhei.ttc"
  ] ++ (if ppocr-native != null then [
    "--prefix" "PATH" ":" "${ppocr-native}/bin"
    "--set-default" "PPOCR_NATIVE_PLANS" "${ppocr-native}/share/ppocr-native"
  ] else []);

  # stage-data.sh lays out a DESTDIR with usr/share; Nix wants $out/share.
  postInstall = ''
    bash $src/tools/stage-data.sh "$TMPDIR/stage"
    mkdir -p "$out/share"
    cp -r "$TMPDIR/stage/usr/share/." "$out/share/"
  '';

  meta = {
    description = "Make scanned PDFs searchable with OCR";
    homepage = "https://github.com/biglinux/bigocrpdf";
    license = with lib.licenses; [ gpl3Plus cc-by-sa-30 lgpl3Only lgpl3Plus ];
    mainProgram = "bigocrpdf";
    platforms = lib.platforms.linux;
  };
}
