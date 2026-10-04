#!/usr/bin/env bash
# Clone the four earlier projects into their helper directories, each at the commit we
# audited. They were git submodules until this repository stopped carrying a .git of its
# own; the pins are the same.
#
# None of their files belong to us and none are committed here. Re-run this after a fresh
# clone, before any of the prior-art runners.
#
#   ./clone_prior_art.sh            clone or update all four
#   ./clone_prior_art.sh --check    print what is present and at which commit

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# helper directory | upstream | pinned commit
PROJECTS=(
  "maria-helper/BR41N.IO-Hackathon-ECoG-Epilepsy|https://github.com/MariaGoniIba/BR41N.IO-Hackathon-ECoG-Epilepsy.git|e2e61599edf80693f2a4cd027771ecc4f7da129a"
  "neverix-helper/ecog-br41n-2022|https://github.com/neverix/ecog-br41n-2022.git|4833d1787dad0f1fe047bf5fe2bc7b4a1cad4925"
  "eidantz-helper/br41n.io_Hackathon_ECoG|https://github.com/Eidantz/br41n.io_Hackathon_ECoG.git|e1b6f6aa7aa0f1a1c0635f55a9adf1f1ee1adccb"
  "bkowshik-helper/br41n-ecog-hand-pose|https://github.com/bkowshik/br41n-ecog-hand-pose.git|41ad6d2de0cac7038b3a1ee6f6c899e79ff10cce"
)

if [ "${1:-}" = "--check" ]; then
  for entry in "${PROJECTS[@]}"; do
    IFS='|' read -r path url commit <<< "$entry"
    dir="$ROOT/prior-art/$path"
    if [ -d "$dir/.git" ]; then
      at=$(git -C "$dir" rev-parse HEAD)
      [ "$at" = "$commit" ] && state="at the pin" || state="at $at, NOT the pin"
      printf '%-44s %s\n' "$path" "$state"
    else
      printf '%-44s %s\n' "$path" "not cloned"
    fi
  done
  exit 0
fi

# bkowshik keeps a 123 MB copy of the recording in git-lfs. Skip it here: fetch_dataset.sh
# pulls it only when the official download is unavailable.
export GIT_LFS_SKIP_SMUDGE=1

status=0
for entry in "${PROJECTS[@]}"; do
  IFS='|' read -r path url commit <<< "$entry"
  dir="$ROOT/prior-art/$path"
  if [ -d "$dir/.git" ]; then
    echo "updating $path"
    git -C "$dir" fetch -q origin "$commit" 2>/dev/null || git -C "$dir" fetch -q origin
  else
    echo "cloning $path"
    git clone -q "$url" "$dir" || { echo "  clone failed: $url" >&2; status=1; continue; }
  fi
  git -C "$dir" checkout -q "$commit" || { echo "  no commit $commit in $url" >&2; status=1; }
done
exit $status
