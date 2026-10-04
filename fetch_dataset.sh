#!/usr/bin/env bash
# Put ECoG_Handpose.mat in the repository root, and check it against the sha256 every
# number here was computed on.
#
# The official archive comes first. If that download or the unpacker is unavailable, the
# same file is pulled out of the bkowshik clone's git-lfs store, which needs only git-lfs.
# Both paths end at the same bytes.
#
#   ./fetch_dataset.sh             official archive, then the local clone
#   ./fetch_dataset.sh --local     skip the download and use the clone

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="$ROOT/ECoG_Handpose.mat"
SUB="$ROOT/prior-art/bkowshik-helper/br41n-ecog-hand-pose"
URL="https://www.gtec.at/downloads_QyTs23/Hackathon/ecog-hand-pose.rar"
SHA256="b2b65f0f040ee9663c523801d92ba7e685d2ade94a2c1592c9975320966fab60"
BYTES=123246843

verify () {
  local got_bytes got_sha
  got_bytes=$(stat -c %s "$1")
  got_sha=$(sha256sum "$1" | cut -d' ' -f1)
  if [ "$got_bytes" != "$BYTES" ] || [ "$got_sha" != "$SHA256" ]; then
    echo "CHECKSUM MISMATCH for $1" >&2
    echo "  expected $BYTES bytes, sha256 $SHA256" >&2
    echo "  got      $got_bytes bytes, sha256 $got_sha" >&2
    return 1
  fi
  echo "ok: $1 ($got_bytes bytes, sha256 verified)"
}

from_official () {
  local unpack tmp
  for candidate in unar unrar 7z; do
    command -v "$candidate" >/dev/null && unpack="$candidate" && break
  done
  [ -n "${unpack:-}" ] || { echo "no RAR5 unpacker (unar, unrar or 7z)" >&2; return 1; }
  command -v curl >/dev/null || { echo "curl is not installed" >&2; return 1; }

  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' RETURN
  echo "downloading the official archive (115 MB)..."
  curl -fsSL -o "$tmp/ecog-hand-pose.rar" "$URL" || return 1
  case "$unpack" in
    unar)  unar -q -o "$tmp" "$tmp/ecog-hand-pose.rar" >/dev/null ;;
    unrar) unrar x -inul "$tmp/ecog-hand-pose.rar" "$tmp/" ;;
    7z)    7z x -y -o"$tmp" "$tmp/ecog-hand-pose.rar" >/dev/null ;;
  esac
  local found
  found=$(find "$tmp" -name ECoG_Handpose.mat -print -quit)
  [ -n "$found" ] || { echo "the archive held no ECoG_Handpose.mat" >&2; return 1; }
  cp "$found" "$DST"
  # data_description.pdf and Hand.pdf travel in the same archive if you want them
  verify "$DST"
}

from_local_clone () {
  command -v git-lfs >/dev/null || { echo "git-lfs is not installed" >&2; return 1; }
  [ -e "$SUB/.git" ] || {
    echo "the bkowshik project is not cloned yet; run ./clone_prior_art.sh first" >&2
    return 1
  }
  echo "pulling from the bkowshik clone's git-lfs store (123 MB)..."
  git -C "$SUB" lfs pull --include data/ECoG_Handpose.mat
  [ -f "$SUB/data/ECoG_Handpose.mat" ] || { echo "lfs pull produced no file" >&2; return 1; }
  cp "$SUB/data/ECoG_Handpose.mat" "$DST"
  verify "$DST"
}

if [ "${1:-}" = "--local" ]; then
  from_local_clone
elif ! from_official; then
  echo "the official archive did not work out; falling back to the local clone" >&2
  from_local_clone
fi
