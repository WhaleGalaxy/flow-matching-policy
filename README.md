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

## Environment setup

```bash
conda activate robot          # python 3.10, torch 2.5.1+cu121
```

### GPU (this machine)

The full `nvidia-driver-570` metapackage breaks the GUI on this laptop
(Intel iGPU + RTX 4060 hybrid, Ubuntu 20.04 + Xorg 1.20): it installs an Xorg
display driver and replaces the system GL stack, and gdm then fails to start.

Only the compute half is needed. `scripts/setup_gpu_headless.sh` installs
`nvidia-headless-570` + `nvidia-utils-570` (kernel module + CUDA runtime, no
Xorg driver, no GL libs) and configures the modules to *not* auto-load at boot,
so the boot path is untouched:

```bash
sudo bash scripts/setup_gpu_headless.sh
```

Then, before any training run:

```bash
sudo modprobe nvidia nvidia_uvm    # load the module on demand
nvidia-smi                          # verify
```

Recovery, if the display ever breaks again:

```bash
sudo apt purge '^nvidia-.*' && sudo apt autoremove
```
