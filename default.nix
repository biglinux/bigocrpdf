{
  python3Packages,
  gtk4,
  libadwaita,
  pkg-config,
  wrapGAppsHook4,
  gobject-introspection,
  tesseract,
  ocrmypdf,
  poppler_utils,
}:

python3Packages.buildPythonApplication {
  pname = "bigocrpdf";
  version = "2.0.0";

  src = ./.;

  pyproject = true;

  build-system = with python3Packages; [ setuptools wheel ];
  
  dependencies = with python3Packages; [
    pygobject3
    pycairo
    ocrmypdf
  ];

  nativeBuildInputs = [
    pkg-config
    wrapGAppsHook4
    gobject-introspection
  ];
  
  buildInputs = [
    gtk4
    libadwaita
    tesseract
    ocrmypdf
    poppler_utils  # For pdfinfo
  ];

  postInstall = ''
    # Install desktop file and icons
    mkdir -p $out/share/applications
    mkdir -p $out/share/icons/hicolor/scalable/apps
    
    cp $src/bigocrpdf/usr/share/applications/*.desktop $out/share/applications/ || true
    cp -r $src/bigocrpdf/usr/share/icons/hicolor/* $out/share/icons/hicolor/ || true
    
    # Install KDE service menu
    mkdir -p $out/share/kio/servicemenus
    cp $src/bigocrpdf/usr/share/kio/servicemenus/*.desktop $out/share/kio/servicemenus/ || true
    
    # Install locale files
    cp -r $src/bigocrpdf/usr/share/locale $out/share/ || true
  '';

  meta = {
    description = "Add OCR to your PDF documents to make them searchable";
    homepage = "https://github.com/biglinux/bigocrpdf";
    license = "GPL-3.0";
    mainProgram = "bigocrpdf";
  };
}
