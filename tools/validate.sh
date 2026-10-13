#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo_root"

workdir=$(mktemp -d)
trap 'rm -rf -- "$workdir"' EXIT
export LC_ALL=C
# LC_ALL alone is not enough to get untranslated output from a Python program.
# GNU gettext ignores LANGUAGE when the locale is C; Python's gettext.find()
# does not, so it would still load the developer's own catalog, and the CLI
# compatibility baseline, a snapshot of English output, would fail on any
# translated desktop.
export LANGUAGE=C
export PYTHONPYCACHEPREFIX="$workdir/pycache"

python3 -m compileall -q src tools usr/share/nautilus-python/extensions
ruff check .
ruff format --check .
pyright
python3 tools/compatibility_verifier.py verify \
	--source-root . \
	--baseline tools/compatibility-baseline.json

version=$(python3 -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
mapfile -d '' sources < <(find src/bigocrpdf usr/share/nautilus-python/extensions \
	-type f -name '*.py' -print0 | sort -z)
xgettext \
	--language=Python \
	--from-code=UTF-8 \
	--keyword=_ \
	--keyword=N_ \
	--keyword=ngettext:1,2 \
	--package-name=bigocrpdf \
	--package-version="$version" \
	--msgid-bugs-address=biglinux@biglinux.com.br \
	--copyright-holder='BigLinux Team' \
	--output="$workdir/bigocrpdf.pot" \
	"${sources[@]}"
diff -u \
	<(sed '/POT-Creation-Date:/d' locale/bigocrpdf.pot) \
	<(sed '/POT-Creation-Date:/d' "$workdir/bigocrpdf.pot")

# msgfmt's exit status flags real defects, and loading each catalog the way the
# application does catches the unfilled Plural-Forms header msgfmt accepts.
for catalog in locale/*.po; do
	msgfmt --check --output-file="$workdir/catalog.mo" "$catalog" || {
		printf 'gettext: %s has a fatal defect (see above)\n' "$catalog" >&2
		exit 1
	}
	CATALOG="$catalog" python3 - "$workdir/catalog.mo" <<-'PY' || exit 1
		import gettext
		import os
		import sys

		catalog = os.environ["CATALOG"]
		try:
		    with open(sys.argv[1], "rb") as handle:
		        translations = gettext.GNUTranslations(handle)
		    # Resolving a plural evaluates the Plural-Forms expression, which is
		    # where an unfilled or malformed header raises.
		    translations.ngettext("one", "many", 2)
		except Exception as exc:
		    print(f"gettext: {catalog} cannot be loaded by Python: {exc}", file=sys.stderr)
		    raise SystemExit(1) from None
	PY
	for state in untranslated fuzzy obsolete; do
		case $state in
		untranslated) options=(--untranslated --no-obsolete) ;;
		fuzzy) options=(--fuzzy --no-obsolete) ;;
		obsolete) options=(--obsolete) ;;
		esac
		stats=$(msgattrib "${options[@]}" "$catalog" | msgfmt --statistics -o /dev/null - 2>&1)
		if [[ $stats != '0 translated messages.' ]]; then
			printf 'gettext state %s remains in %s: %s\n' "$state" "$catalog" "$stats" >&2
			exit 1
		fi
	done
done

if bash tools/stage-data.sh // >/dev/null 2>&1; then
	printf 'stage-data.sh accepted an unsafe root DESTDIR\n' >&2
	exit 1
fi
ln -s / "$workdir/root-link"
if bash tools/stage-data.sh "$workdir/root-link" >/dev/null 2>&1; then
	printf 'stage-data.sh accepted a DESTDIR resolving to root\n' >&2
	exit 1
fi

bash tools/stage-data.sh "$workdir/stage"
# stage-data.sh leaves bytecode out of the stage, so a __pycache__ that running
# the application from the checkout leaves under usr/ is not compared.
while IFS= read -r -d '' file; do
	cmp "$file" "$workdir/stage/$file"
done < <(find usr/share -type f -not -name '*.py[co]' -not -path '*/__pycache__/*' -print0)

desktop-file-validate "$workdir"/stage/usr/share/applications/*.desktop
appstreamcli validate --no-net --pedantic "$workdir"/stage/usr/share/metainfo/*.metainfo.xml
bash -n pkgbuild/PKGBUILD tools/stage-data.sh tools/validate.sh tools/quality.sh tools/benchmark.sh
shellcheck pkgbuild/PKGBUILD tools/stage-data.sh tools/validate.sh tools/quality.sh tools/benchmark.sh
shfmt -d pkgbuild/PKGBUILD tools/stage-data.sh tools/validate.sh tools/quality.sh tools/benchmark.sh
python3 -m pytest -q -p no:cacheprovider

# The opt-in layers are excluded from the run above, so nothing would notice a
# marker typo there. Collecting them under --strict-markers does.
python3 -m pytest -q -p no:cacheprovider --collect-only \
	-m 'slow or real_ocr or benchmark or network' >/dev/null

printf 'Validation passed.\n'
