#!/usr/bin/env bash
# Fetch the original GPT-1 checkpoint (MIT-licensed, ~470 MB) straight from OpenAI's 2018 repo.
# No Hugging Face account or hub access required.
set -euo pipefail
DEST="${1:-$HOME/.cache/gpt1}"
if [ ! -f "$DEST/model/params_9.npy" ]; then
  git clone --depth 1 https://github.com/openai/finetune-transformer-lm.git "$DEST"
fi
echo "GPT-1 weights in $DEST/model"
echo "export GPT1_DIR=$DEST/model"
