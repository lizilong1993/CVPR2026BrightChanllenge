# CVPR 2026 BRIGHT Challenge Starter Repository

This repository is organized for participating in the CVPR 2026 BRIGHT Challenge:
"Advancing All-Weather Building Damage Mapping to Instance-Level".

The primary working directory is [`cvprw26/`](cvprw26/README.md). Other folders are kept as supporting references, legacy baselines, or related BRIGHT benchmarks.

## Main entry point

Start from [`cvprw26/README.md`](cvprw26/README.md).

That subproject contains:

- the Mask R-CNN baseline
- training and inference code
- split files and annotation manifests
- submission file generation for Codabench

Useful files:

- [`cvprw26/config/disaster.yaml`](cvprw26/config/disaster.yaml)
- [`cvprw26/src/train.py`](cvprw26/src/train.py)
- [`cvprw26/src/infer.py`](cvprw26/src/infer.py)
- [`cvprw26/tools/merge_coco_json.py`](cvprw26/tools/merge_coco_json.py)

## Quick start

```bash
cd cvprw26
conda create -n bright_cvprw26 python=3.10 -y
conda activate bright_cvprw26
pip install -e .
```

Prepare the BRIGHT challenge data and make sure it follows:

```text
<BRIGHT_ROOT>/
- pre-event/
- post-event/
- target_instance_level/
```

If merged COCO files are not already present, generate them:

```bash
python tools/merge_coco_json.py \
  --json-dir <BRIGHT_ROOT>/target_instance_level \
  --image-dir <BRIGHT_ROOT>/post-event \
  --pre-event-dir <BRIGHT_ROOT>/pre-event \
  --splits-dir data/splits \
  --output-dir data/instance_annotations
```

Then set paths in [`cvprw26/config/disaster.yaml`](cvprw26/config/disaster.yaml) and run:

```bash
python -m src.train --config config/disaster.yaml
python -m src.infer --config config/disaster.yaml
```

## Challenge resources

- Challenge README: [`cvprw26/README.md`](cvprw26/README.md)
- Challenge page: [MONTI / CVPR 2026 Workshop](https://sites.google.com/view/monti2026/home)
- Submission server: [Codabench competition](https://www.codabench.org/competitions/15134/)
- Dataset: [Zenodo](https://zenodo.org/records/14619797)
- Dataset mirror: [HuggingFace](https://huggingface.co/datasets/Kullervo/BRIGHT)

Some optical data over Ukraine, Myanmar, and Mexico cannot be redistributed directly. See [`tutorial.md`](tutorial.md) for preprocessing instructions.

## Repository layout

- [`cvprw26/`](cvprw26/README.md): main challenge codebase for training, inference, and submission
- [`bda_benchmark/`](bda_benchmark/README_cross_event.md): related BRIGHT building-damage benchmarks
- [`umcd_benchmark/`](umcd_benchmark/README.md): unsupervised multimodal change detection benchmark
- [`umim_benchmark/`](umim_benchmark/README.md): unsupervised multimodal image matching benchmark
- [`dfc25_legacy/`](dfc25_legacy): legacy DFC 2025 code kept for reference
- [`tutorial.md`](tutorial.md): optical image download and georeferencing workflow

## Citation

If this repository or the BRIGHT dataset helps your work, please cite:

```bibtex
@Article{Chen2025Bright,
    AUTHOR = {Chen, H. and Song, J. and Dietrich, O. and Broni-Bediako, C. and Xuan, W. and Wang, J. and Shao, X. and Wei, Y. and Xia, J. and Lan, C. and Schindler, K. and Yokoya, N.},
    TITLE = {\textsc{Bright}: a globally distributed multimodal building damage assessment dataset with very-high-resolution for all-weather disaster response},
    JOURNAL = {Earth System Science Data},
    VOLUME = {17},
    YEAR = {2025},
    NUMBER = {11},
    PAGES = {6217--6253},
    DOI = {10.5194/essd-17-6217-2025}
}
```
