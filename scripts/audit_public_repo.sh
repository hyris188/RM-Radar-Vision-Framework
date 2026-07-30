#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

forbidden_pattern='^(weights|checkpoints|datasets?|data|runs|logs|backups|recordings|rejected)/|(^|/)(weights|checkpoints)/|\.(pt|pth|onnx|engine|ckpt|safetensors|trt|jpg|jpeg|png|bmp|webp|gif|mp4|avi|mkv|tar|tgz|zip|7z)$|\.tar\.gz$'
forbidden="$(git ls-files | grep -Ei "$forbidden_pattern" || true)"

if [[ -n "$forbidden" ]]; then
  echo "ERROR: prohibited tracked files:" >&2
  echo "$forbidden" >&2
  exit 1
fi

if git grep -En '/home/|/localdata/|github_pat_|ghp_|AKIA[0-9A-Z]{16}|BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY' -- ':!scripts/audit_public_repo.sh'; then
  echo "ERROR: local paths or likely secrets found in tracked text." >&2
  exit 1
fi

echo "Public repository audit passed: no tracked weights, datasets, media, archives, local paths or obvious secrets."
