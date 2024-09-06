#!/bin/bash
  CUDA_VISIBLE_DEVICES=0 python dynamic_shot_capacity_adjustment.py     --config configs \
                                                --wandb-log \
                                                --datasets A/V/R/S \
                                                --backbone RN50