#!/bin/bash
set -eou pipefail

FIG="${1:-}"
if [ -z "$FIG" ]; then
    echo "usage: figure.sh <num>"
    exit 1
fi

cat << EOF
==> Checking availability of source data"
NOTE: Availability of data_*.nc files are restricted due to
      license, please contact DART-Pipeline authors to get access

EOF

if [ "$(uname)" == "Darwin" ]; then
    shasum -a 256 -c SHA256SUMS.txt
else
    sha256sum -c SHA256SUMS.txt
fi

echo "==> Generating figure $FIG"
case "$FIG" in
    "1")
        echo This figure is not generated from data
        exit 0
        ;;
    "2")
        uv run python Fig2.py
        ;;
    "3")
        uv run python Fig3.py
        ;;
esac
