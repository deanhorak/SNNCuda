#!/usr/bin/env bash
set -euo pipefail

if ! command -v clang-format >/dev/null 2>&1; then
  echo "clang-format not found" >&2
  exit 1
fi

find include src examples tests -type f \( -name '*.h' -o -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' \) -print0 \
  | xargs -0 clang-format -i

