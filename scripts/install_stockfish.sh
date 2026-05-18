#!/usr/bin/env bash
set -euo pipefail

VERSION="${VERSION:-sf_18}"
ASSET="${ASSET:-stockfish-ubuntu-x86-64.tar}"
BASE_URL="${BASE_URL:-https://github.com/official-stockfish/Stockfish/releases/download}"
DEST="${DEST:-tools/stockfish}"
ARCH="${ARCH:-x86-64}"

mkdir -p "$DEST"
archive="$DEST/$ASSET"
url="$BASE_URL/$VERSION/$ASSET"

curl -L "$url" -o "$archive"
tar -xf "$archive" -C "$DEST"

binary=""
if [[ "${USE_RELEASE_BINARY:-0}" != "1" && -f "$DEST/stockfish/src/Makefile" ]]; then
  make -C "$DEST/stockfish/src" -j"$(nproc)" build "ARCH=$ARCH"
  binary="$DEST/stockfish/src/stockfish"
fi
if [[ -z "$binary" ]]; then
  binary="$(find "$DEST" -type f -name 'stockfish*' -perm -111 | head -1)"
fi
if [[ -z "$binary" && -f "$DEST/stockfish/src/Makefile" ]]; then
  make -C "$DEST/stockfish/src" -j"$(nproc)" build "ARCH=$ARCH"
  binary="$DEST/stockfish/src/stockfish"
fi

if [[ -z "$binary" ]]; then
  echo "No executable stockfish binary found under $DEST" >&2
  exit 1
fi

mkdir -p "$DEST/bin"
ln -sf "$(realpath --relative-to="$DEST/bin" "$binary")" "$DEST/bin/stockfish"
"$DEST/bin/stockfish" bench 1 >/dev/null
echo "$DEST/bin/stockfish"
