set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FILE=hammerrank-runtime-v4.18.0.tar.gz
SHA=6dfc105be9e42dd211d43dde56d53f04b3d4a056c2e17096523f331f2f51be17
URL="${HAMMERRANK_ARTIFACTS_URL:-https://github.com/HansleCho/hammerrank/releases/download/v0.2.0/$FILE}"
cd "$ROOT"
if [ ! -f "$FILE" ]; then
  echo "downloading $URL"
  curl -L --fail --progress-bar -o "$FILE.part" "$URL" || { echo "download failed; if the release is hosted elsewhere, set HAMMERRANK_ARTIFACTS_URL or place $FILE in $ROOT"; exit 1; }
  mv "$FILE.part" "$FILE"
fi
if command -v shasum >/dev/null 2>&1; then got=$(shasum -a 256 "$FILE" | cut -d' ' -f1); else got=$(sha256sum "$FILE" | cut -d' ' -f1); fi
if [ "$got" != "$SHA" ]; then echo "checksum mismatch: $got != $SHA"; exit 1; fi
echo "extracting into $ROOT/data"
tar -xzf "$FILE" && rm -f "$FILE"
ls -la data
