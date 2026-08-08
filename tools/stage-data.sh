#!/usr/bin/env bash
set -euo pipefail

if (($# != 1)); then
	printf 'Usage: %s DESTDIR\n' "${0##*/}" >&2
	exit 2
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)

if [[ -z $1 ]]; then
	printf 'Refusing unsafe DESTDIR: %s\n' "$1" >&2
	exit 2
fi

destdir=$(realpath -m -- "$1")
if [[ $destdir == / ]]; then
	printf 'Refusing unsafe DESTDIR: %s\n' "$1" >&2
	exit 2
fi

install -d "$destdir/usr/share"
cp -a --no-preserve=ownership "$repo_root/usr/share/." "$destdir/usr/share/"
find "$destdir/usr/share" -type f -name '*.py[co]' -delete
find "$destdir/usr/share" -depth -type d -name __pycache__ -empty -delete

for catalog in "$repo_root"/locale/*.po; do
	language=${catalog##*/}
	language=${language%.po}
	output="$destdir/usr/share/locale/$language/LC_MESSAGES/bigocrpdf.mo"
	install -d "${output%/*}"
	msgfmt --check --output-file="$output" "$catalog"
done

install -Dm644 "$repo_root/LICENSE" \
	"$destdir/usr/share/licenses/bigocrpdf/LICENSE"
install -Dm644 "$repo_root/README.md" \
	"$destdir/usr/share/doc/bigocrpdf/README.md"
