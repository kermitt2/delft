#!/usr/bin/env bash
# Build the wheel and the sdist of DeLFT, check them, and install the wheel in a fresh virtual environment
# where every application has to start, run from outside the checkout so that nothing is imported from it.
#
# usage: scripts/check_wheel.sh [dist directory, default dist]
#
# The checks: twine accepts both archives, the wheel holds the delft package only (no tests package next to
# it), the version of the archives is the one of pyproject.toml, and `python -m delft.applications.<name> --help`
# exits with 0 for every application.
set -euo pipefail

DIST="${1:-dist}"
PYTHON="${PYTHON:-python}"
CHECKOUT="$(cd "$(dirname "$0")/.." && pwd)"

rm -rf "${DIST:?}"
"$PYTHON" -m build --outdir "$DIST" "$CHECKOUT"
"$PYTHON" -m twine check --strict "$DIST"/*

WHEEL="$(ls "$DIST"/*.whl)"
VERSION="$(sed -n 's/^version = "\(.*\)"$/\1/p' "$CHECKOUT/pyproject.toml" | head -1)"
case "$(basename "$WHEEL")" in
    delft-"$VERSION"-*) ;;
    *) echo "the wheel $WHEEL is not the version $VERSION of pyproject.toml" >&2; exit 1 ;;
esac

TOP_LEVEL="$("$PYTHON" -c "
import sys, zipfile
names = {name.split('/')[0] for name in zipfile.ZipFile(sys.argv[1]).namelist()}
print(' '.join(sorted(name for name in names if not name.endswith('.dist-info'))))
" "$WHEEL")"
if [ "$TOP_LEVEL" != "delft" ]; then
    echo "the wheel holds $TOP_LEVEL, not the delft package alone" >&2
    exit 1
fi

VENV="$(mktemp -d)/venv"
"$PYTHON" -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
# the CPU build of torch: the CUDA one of PyPI weighs several gigabytes
"$VENV/bin/python" -m pip install --quiet --extra-index-url https://download.pytorch.org/whl/cpu "$WHEEL"

cd "$(mktemp -d)"
"$VENV/bin/python" -c "import delft, delft.sequenceLabelling, delft.textClassification; print('delft imported from', delft.__file__)"
for application in "$VENV"/lib/python*/site-packages/delft/applications/*.py; do
    name="$(basename "$application" .py)"
    [ "$name" = "__init__" ] && continue
    echo "--- $name --help"
    "$VENV/bin/python" -m "delft.applications.$name" --help > /dev/null
done
echo "the wheel $WHEEL installs and its applications start"
