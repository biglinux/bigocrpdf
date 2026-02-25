"""Nautilus extension for BigOCR PDF / BigOCR Image context menu actions."""

import locale
import os
import subprocess

from gi.repository import GObject, Nautilus

# Image MIME types supported by the application
_IMAGE_MIMES = frozenset((
    "image/avif",
    "image/gif",
    "image/heif",
    "image/jpeg",
    "image/jxl",
    "image/png",
    "image/bmp",
    "image/x-eps",
    "image/x-icns",
    "image/x-ico",
    "image/x-portable-bitmap",
    "image/x-portable-graymap",
    "image/x-portable-pixmap",
    "image/x-xbitmap",
    "image/x-xpixmap",
    "image/tiff",
    "image/x-psd",
    "image/x-webp",
    "image/webp",
    "image/x-tga",
    "image/x-kde-raw",
    "image/x-canon-cr2",
    "image/x-canon-crw",
    "image/x-kodak-dcr",
    "image/x-adobe-dng",
    "image/x-kodak-k25",
    "image/x-kodak-kdc",
    "image/x-minolta-mrw",
    "image/x-nikon-nef",
    "image/x-olympus-orf",
    "image/x-pentax-pef",
    "image/x-fuji-raf",
    "image/x-panasonic-rw",
    "image/x-sony-sr2",
    "image/x-sony-srf",
    "image/x-sigma-x3f",
    "image/x-sony-arw",
    "image/x-panasonic-rw2",
))

_PDF_MIME = "application/pdf"

# Translations keyed by language code
_TRANSLATIONS = {
    "pt": {
        "ocr_pdf": "Reconhecimento de texto (OCR)",
        "edit_pdf": "Editar páginas",
        "ocr_image": "Reconhecimento de texto (OCR)",
        "create_pdf": "Criar PDF",
        "export_pdf": "Exportar para ODT ou TXT",
    },
    "bg": {
        "ocr_pdf": "Разпознаване на текст (OCR)",
        "edit_pdf": "Редактиране на страници",
        "ocr_image": "Разпознаване на текст (OCR)",
        "create_pdf": "Създай PDF",
        "export_pdf": "Експорт в ODT или TXT",
    },
    "cs": {
        "ocr_pdf": "Rozpoznávání textu (OCR)",
        "edit_pdf": "Upravit stránky",
        "ocr_image": "Rozpoznávání textu (OCR)",
        "create_pdf": "Vytvořit PDF",
        "export_pdf": "Exportovat do ODT nebo TXT",
    },
    "da": {
        "ocr_pdf": "Tekstgenkendelse (OCR)",
        "edit_pdf": "Rediger sider",
        "ocr_image": "Tekstgenkendelse (OCR)",
        "create_pdf": "Opret PDF",
        "export_pdf": "Eksportér til ODT eller TXT",
    },
    "de": {
        "ocr_pdf": "Texterkennung (OCR)",
        "edit_pdf": "Seiten bearbeiten",
        "ocr_image": "Texterkennung (OCR)",
        "create_pdf": "PDF erstellen",
        "export_pdf": "Exportieren nach ODT oder TXT",
    },
    "el": {
        "ocr_pdf": "Αναγνώριση κειμένου (OCR)",
        "edit_pdf": "Επεξεργασία σελίδων",
        "ocr_image": "Αναγνώριση κειμένου (OCR)",
        "create_pdf": "Δημιουργία PDF",
        "export_pdf": "Εξαγωγή σε ODT ή TXT",
    },
    "es": {
        "ocr_pdf": "Reconocimiento de texto (OCR)",
        "edit_pdf": "Editar páginas",
        "ocr_image": "Reconocimiento de texto (OCR)",
        "create_pdf": "Crear PDF",
        "export_pdf": "Exportar a ODT o TXT",
    },
    "et": {
        "ocr_pdf": "Tekstituvastus (OCR)",
        "edit_pdf": "Muuda lehekülgi",
        "ocr_image": "Tekstituvastus (OCR)",
        "create_pdf": "Loo PDF",
        "export_pdf": "Ekspordi ODT- või TXT-vormingusse",
    },
    "fi": {
        "ocr_pdf": "Tekstintunnistus (OCR)",
        "edit_pdf": "Muokkaa sivuja",
        "ocr_image": "Tekstintunnistus (OCR)",
        "create_pdf": "Luo PDF",
        "export_pdf": "Vie ODT- tai TXT-muotoon",
    },
    "fr": {
        "ocr_pdf": "Reconnaissance de texte (OCR)",
        "edit_pdf": "Modifier les pages",
        "ocr_image": "Reconnaissance de texte (OCR)",
        "create_pdf": "Créer un PDF",
        "export_pdf": "Exporter en ODT ou TXT",
    },
    "he": {
        "ocr_pdf": "זיהוי טקסט (OCR)",
        "edit_pdf": "עריכת דפים",
        "ocr_image": "זיהוי טקסט (OCR)",
        "create_pdf": "צור PDF",
        "export_pdf": "ייצוא ל-ODT או TXT",
    },
    "hr": {
        "ocr_pdf": "Prepoznavanje teksta (OCR)",
        "edit_pdf": "Uredi stranice",
        "ocr_image": "Prepoznavanje teksta (OCR)",
        "create_pdf": "Stvori PDF",
        "export_pdf": "Izvezi u ODT ili TXT",
    },
    "hu": {
        "ocr_pdf": "Szövegfelismerés (OCR)",
        "edit_pdf": "Oldalak szerkesztése",
        "ocr_image": "Szövegfelismerés (OCR)",
        "create_pdf": "PDF létrehozása",
        "export_pdf": "Exportálás ODT vagy TXT formátumba",
    },
    "is": {
        "ocr_pdf": "Textagreining (OCR)",
        "edit_pdf": "Breyta síðum",
        "ocr_image": "Textagreining (OCR)",
        "create_pdf": "Búa til PDF",
        "export_pdf": "Flytja út í ODT eða TXT",
    },
    "it": {
        "ocr_pdf": "Riconoscimento testo (OCR)",
        "edit_pdf": "Modifica pagine",
        "ocr_image": "Riconoscimento testo (OCR)",
        "create_pdf": "Crea PDF",
        "export_pdf": "Esporta in ODT o TXT",
    },
    "ja": {
        "ocr_pdf": "テキスト認識 (OCR)",
        "edit_pdf": "ページを編集",
        "ocr_image": "テキスト認識 (OCR)",
        "create_pdf": "PDF を作成",
        "export_pdf": "ODT または TXT にエクスポート",
    },
    "ko": {
        "ocr_pdf": "텍스트 인식 (OCR)",
        "edit_pdf": "페이지 편집",
        "ocr_image": "텍스트 인식 (OCR)",
        "create_pdf": "PDF 만들기",
        "export_pdf": "ODT 또는 TXT로 내보내기",
    },
    "nl": {
        "ocr_pdf": "Tekstherkenning (OCR)",
        "edit_pdf": "Pagina's bewerken",
        "ocr_image": "Tekstherkenning (OCR)",
        "create_pdf": "PDF maken",
        "export_pdf": "Exporteren naar ODT of TXT",
    },
    "no": {
        "ocr_pdf": "Tekstgjenkjenning (OCR)",
        "edit_pdf": "Rediger sider",
        "ocr_image": "Tekstgjenkjenning (OCR)",
        "create_pdf": "Lag PDF",
        "export_pdf": "Eksporter til ODT eller TXT",
    },
    "pl": {
        "ocr_pdf": "Rozpoznawanie tekstu (OCR)",
        "edit_pdf": "Edytuj strony",
        "ocr_image": "Rozpoznawanie tekstu (OCR)",
        "create_pdf": "Utwórz PDF",
        "export_pdf": "Eksportuj do ODT lub TXT",
    },
    "ro": {
        "ocr_pdf": "Recunoaștere text (OCR)",
        "edit_pdf": "Editează paginile",
        "ocr_image": "Recunoaștere text (OCR)",
        "create_pdf": "Creează PDF",
        "export_pdf": "Exportă în ODT sau TXT",
    },
    "ru": {
        "ocr_pdf": "Распознавание текста (OCR)",
        "edit_pdf": "Редактировать страницы",
        "ocr_image": "Распознавание текста (OCR)",
        "create_pdf": "Создать PDF",
        "export_pdf": "Экспорт в ODT или TXT",
    },
    "sk": {
        "ocr_pdf": "Rozpoznávanie textu (OCR)",
        "edit_pdf": "Upraviť stránky",
        "ocr_image": "Rozpoznávanie textu (OCR)",
        "create_pdf": "Vytvoriť PDF",
        "export_pdf": "Exportovať do ODT alebo TXT",
    },
    "sv": {
        "ocr_pdf": "Textigenkänning (OCR)",
        "edit_pdf": "Redigera sidor",
        "ocr_image": "Textigenkänning (OCR)",
        "create_pdf": "Skapa PDF",
        "export_pdf": "Exportera till ODT eller TXT",
    },
    "tr": {
        "ocr_pdf": "Metin tanıma (OCR)",
        "edit_pdf": "Sayfaları düzenle",
        "ocr_image": "Metin tanıma (OCR)",
        "create_pdf": "PDF oluştur",
        "export_pdf": "ODT veya TXT olarak dışa aktar",
    },
    "uk": {
        "ocr_pdf": "Розпізнавання тексту (OCR)",
        "edit_pdf": "Редагувати сторінки",
        "ocr_image": "Розпізнавання тексту (OCR)",
        "create_pdf": "Створити PDF",
        "export_pdf": "Експорт в ODT або TXT",
    },
    "zh": {
        "ocr_pdf": "文字识别（OCR）",
        "edit_pdf": "编辑页面",
        "ocr_image": "文字识别（OCR）",
        "create_pdf": "创建 PDF",
        "export_pdf": "导出为 ODT 或 TXT",
    },
}

_DEFAULTS = {
    "ocr_pdf": "Text recognition (OCR)",
    "edit_pdf": "Edit pages",
    "ocr_image": "Text recognition (OCR)",
    "create_pdf": "Create PDF",
    "export_pdf": "Export to ODT or TXT",
}


def _get_lang():
    """Return the two-letter language code from the current locale."""
    try:
        loc = locale.getlocale()[0] or os.environ.get("LANG", "en")
    except ValueError:
        loc = os.environ.get("LANG", "en")
    return loc.split("_")[0].split(".")[0]


def _tr(key):
    """Translate *key* to the current locale, falling back to English."""
    lang = _get_lang()
    return _TRANSLATIONS.get(lang, {}).get(key, _DEFAULTS[key])


def _run(cmd, files):
    """Launch a command with the given files in the background."""
    paths = [f.get_location().get_path() for f in files if f.get_location()]
    if not paths:
        return
    subprocess.Popen([cmd] + paths)


class BigOcrPdfExtension(GObject.GObject, Nautilus.MenuProvider):
    """Context menu entries for BigOCR PDF and BigOCR Image."""

    def get_file_items(self, *args):
        # Nautilus 43+ passes (files,); older versions pass (window, files)
        files = args[-1] if args else []
        if not files:
            return []

        items = []
        mimes = {f.get_mime_type() for f in files}

        has_pdf = _PDF_MIME in mimes
        has_image = bool(mimes & _IMAGE_MIMES)

        if has_pdf and not has_image:
            # Pure PDF selection
            item_ocr = Nautilus.MenuItem(
                name="BigOcrPdf::ocr_pdf",
                label=_tr("ocr_pdf"),
                icon="bigocrpdf",
            )
            item_ocr.connect("activate", self._on_ocr_pdf, files)
            items.append(item_ocr)

            item_edit = Nautilus.MenuItem(
                name="BigOcrPdf::edit_pdf",
                label=_tr("edit_pdf"),
                icon="bigocrpdf",
            )
            item_edit.connect("activate", self._on_edit_pdf, files)
            items.append(item_edit)

            item_export = Nautilus.MenuItem(
                name="BigOcrPdf::export_pdf",
                label=_tr("export_pdf"),
                icon="bigocrpdf",
            )
            item_export.connect("activate", self._on_export_pdf, files)
            items.append(item_export)

        elif has_image and not has_pdf:
            # Pure image selection
            if len(files) == 1:
                item_ocr = Nautilus.MenuItem(
                    name="BigOcrPdf::ocr_image",
                    label=_tr("ocr_image"),
                    icon="bigocrimage",
                )
                item_ocr.connect("activate", self._on_ocr_image, files)
                items.append(item_ocr)

            item_create = Nautilus.MenuItem(
                name="BigOcrPdf::create_pdf",
                label=_tr("create_pdf"),
                icon="bigocrpdf",
            )
            item_create.connect("activate", self._on_create_pdf, files)
            items.append(item_create)

        return items

    # -- callbacks ----------------------------------------------------------

    def _on_ocr_pdf(self, _menu, files):
        _run("bigocrpdf", files)

    def _on_edit_pdf(self, _menu, files):
        paths = [f.get_location().get_path() for f in files if f.get_location()]
        if paths:
            subprocess.Popen(["bigocrpdf", "--edit"] + paths)

    def _on_ocr_image(self, _menu, files):
        _run("bigocrimage", files)

    def _on_create_pdf(self, _menu, files):
        paths = [f.get_location().get_path() for f in files if f.get_location()]
        if paths:
            subprocess.Popen(["bigocrpdf", "--edit"] + paths)

    def _on_export_pdf(self, _menu, files):
        paths = [f.get_location().get_path() for f in files if f.get_location()]
        for path in paths:
            subprocess.Popen(["bigocrpdf-cli", "export-dialog", path])
