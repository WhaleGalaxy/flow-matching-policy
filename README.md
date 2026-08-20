# FM Policy — Multi-Task Flow Matching for Robot Manipulation

> 🚧 Work in progress — following the 4-week execution checklist.
> This README is a stub; the full bilingual writeup (results table, quick start, architecture diagram) is a Week 4 task.

## Idea

A multi-task manipulation policy conditioned on vision (DINOv2), language (SigLIP),
and proprioception, trained with Flow Matching (OT-CFM) instead of DDPM-style
diffusion, evaluated on ManiSkill3 (`PickCube-v1`, `StackCube-v1`, `PegInsertionSide-v1`).

## Project layout

```
fm_policy/
├── configs/
│   ├── train.yaml       # top-level hydra config
│   ├── model/            # bc.yaml / fm.yaml
│   └── task/              # pick.yaml / stack.yaml / peg.yaml
├── src/
│   ├── models/
│   │   ├── encoders.py    # VisualEncoder, LanguageEncoder, ProprioEncoder
│   │   ├── denoiser.py    # SinusoidalPosEmb, AdaLN, CrossAttention, FMDenoiser
│   │   └── policy.py      # BCPolicy, FMPolicy (assembles the above)
│   ├── data/
│   │   └── dataset.py     # ManiskillDataset
│   ├── train.py
│   └── evaluate.py
├── tests/
├── scripts/
│   └── download_demos.sh
└── requirements.txt
```

## Status

See the execution checklist for the day-by-day plan. Progress log will be kept here
once Week 1 (encoders + BC baseline) is done.
