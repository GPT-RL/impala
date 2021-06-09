#! /usr/bin/env bash
rm -rf logs
mkdir -p logs ~/.cache/GPT ~/.cache/huggingface
docker build -t impala .
docker run --rm -it --env-file .env --gpus all\
  impala $@
#  -v "$(pwd)/logs:/tmp/bsuite"\
#  -v "$HOME/.cache/GPT/:/root/.cache/GPT" \
#  -v "$HOME/.cache/huggingface/:/root/.cache/huggingface" \
