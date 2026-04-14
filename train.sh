#!/bin/bash
uv run python -m llm_from_scratch.examples.training.train_cloud \
    --model_size medium \
    --epochs 1 \
    --batch_size 32 \
    --checkpoint_dir checkpoints 
