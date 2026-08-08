#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$repo_root"

workdir=$(mktemp -d)
trap 'rm -rf -- "$workdir"' EXIT
export LC_ALL=C
export PYTHONPYCACHEPREFIX="$workdir/pycache"

python3 -m compileall -q src tools usr/share/nautilus-python/extensions
ruff check .
ruff format --check .
pyright
python3 tools/compatibility_verifier.py verify \
	--source-root . \
	--baseline tools/compatibility-baseline.json

mapfile -d '' sources < <(find src/bigocrpdf usr/share/nautilus-python/extensions \
	-type f -name '*.py' -print0 | sort -z)
xgettext \
	--language=Python \
	--from-code=UTF-8 \
	--keyword=_ \
	--keyword=N_ \
	--keyword=ngettext:1,2 \
	--package-name=bigocrpdf \
	--package-version=3.0.0 \
	--msgid-bugs-address=contact@biglinux.com.br \
	--copyright-holder='BigLinux Team' \
	--output="$workdir/bigocrpdf.pot" \
	"${sources[@]}"
diff -u \
	<(sed '/POT-Creation-Date:/d' locale/bigocrpdf.pot) \
	<(sed '/POT-Creation-Date:/d' "$workdir/bigocrpdf.pot")

for catalog in locale/*.po; do
	report=$(msgfmt --check --verbose --output-file=/dev/null "$catalog" 2>&1) || {
		printf '%s\n' "$report" >&2
		exit 1
	}
	if grep --ignore-case --quiet 'warning:' <<<"$report"; then
		printf 'gettext warning in %s: %s\n' "$catalog" "$report" >&2
		exit 1
	fi
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
while IFS= read -r -d '' file; do
	cmp "$file" "$workdir/stage/$file"
done < <(find usr/share -type f -print0)

desktop-file-validate "$workdir"/stage/usr/share/applications/*.desktop
appstreamcli validate --no-net --pedantic "$workdir"/stage/usr/share/metainfo/*.metainfo.xml
bash -n pkgbuild/PKGBUILD tools/stage-data.sh tools/validate.sh
shellcheck pkgbuild/PKGBUILD tools/stage-data.sh tools/validate.sh
shfmt -d pkgbuild/PKGBUILD tools/stage-data.sh tools/validate.sh
python3 -m pytest -q -p no:cacheprovider

printf 'Validation passed.\n'
