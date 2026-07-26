# PestScope

PestScope is a fine-grained pest reasoning segmentation model designed for agricultural visual question-answering scenarios. Building on the language comprehension capabilities of multimodal large language models, it introduces SAM-style segmentation decoding to locate target pests based on user queries and output segmentation masks.

This repository mainly contains the PestScope model, training and evaluation entry points, data-loading logic for pest reasoning segmentation, and code for the data construction pipeline.

## Main Structure

```text
.
|-- train_ft.py                      # PestScope fine-tuning, validation, and eval_only entry point
|-- requirements.txt                 # Python dependencies
|-- scripts/
|   |-- train_pestscope_seg.sh       # PestScope segmentation training template
|   |-- eval_pestscope_seg.sh        # PestScope segmentation evaluation template
|   `-- merge_lora_weights.py        # LoRA weight merging utility
|-- model/
|   |-- PestScope.py                 # PestScope model
|   |-- SAM/                         # Components related to the SAM grounding encoder
|   `-- llava/                       # LLaVA language model and vision tower adapter code
|-- dataset/
|   |-- dataset.py                   # Collation and mixed-dataset wrappers
|   |-- segm_datasets/Pestseg_ds.py  # PestSeg/ReasonSeg data loading
|-- build_fusion_data/               # PestSeg data construction pipeline
```

## Environment Setup

```bash
conda create -n pestscope python=3.10 -y
conda activate pestscope

# Install PyTorch according to the local CUDA version, then install the project dependencies
pip install -r requirements.txt
```

## Model and Data Preparation

At minimum, the following resources are required for training and evaluation:

- GLaMM initialization weights: specified through `--version`, for example `Model/GLaMM-RefSeg/`.
- SAM ViT-H weights: specified through `--vision_pretrained`, for example `Model/SAM-vit-H/sam_vit_h_4b8939.pth`.
- CLIP vision tower: specified through `--vision-tower`, for example `Model/clip-vit-large-patch14-336/`.
- PestSeg/ReasonSeg annotation directories: specified through `--dataset_dir`, `--image_root_path`, `--reason_seg_data_add`, and `--reason_image_root_path`.

By default, `dataset/segm_datasets/Pestseg_ds.py` reads the following structure:

```text
<dataset_dir>/
`-- reason_seg/
    `-- ReasonSeg/
        |-- train/*.json
        |-- val/*.json
        `-- explanatory/
            |-- train.json
            `-- val.json
```

The image directory is specified by `--image_root_path`. If additional reasoning segmentation data is used, `--reason_seg_data_add` should point to another annotation root directory with the same structure, and `--reason_image_root_path` should point to its image directory.

The pretrained PestScope-7B model is available at:

```text
https://www.modelscope.cn/models/luohuibin/PestScope-7b
```

The PestScope-7B training dataset is available at:

```text
https://www.modelscope.cn/datasets/luohuibin/pestscope_data
```

## Dataset Construction

The dataset construction pipeline is documented separately and is not repeated in the main README. See:

```text
build_fusion_data/DATA_PIPELINE_USAGE.md
```

This document covers preprocessing of the original Pest24 data, SAM2 mask generation, construction of `pest24_database.json`, generation of reason/refer fusion data at three difficulty levels, and the final ReasonSeg/ReferSeg JSON conversion pipeline.

## Training

The PestScope segmentation training template is located at:

```bash
bash scripts/train_pestscope_seg.sh
```

The script invokes `train_ft.py`. An example of the core parameters is shown below. In actual use, replace all local paths with the paths to your own models, data, and weights.

```bash
deepspeed \
  --master_port=24969 \
  --include=localhost:0 \
  train_ft.py \
  --version="/path/to/GLaMM-RefSeg" \
  --dataset_dir="/path/to/refer_data/dif_data" \
  --reason_seg_data_add="/path/to/reason_data/dif_data" \
  --reason_image_root_path="/path/to/images" \
  --image_root_path="/path/to/images" \
  --vision_pretrained="/path/to/sam_vit_h_4b8939.pth" \
  --vision-tower="/path/to/clip-vit-large-patch14-336" \
  --exp_name="PestScope-7b" \
  --lora_r=8 \
  --lr=0.0001 \
  --pretrained \
  --use_segm_data \
  --seg_dataset="Refer_Segm" \
  --segm_sample_rates="1" \
  --refer_segm_data="refcocog" \
  --val_dataset="RefCOCOgSegmVal" \
  --epochs=10 \
  --steps_per_epoch=1208 \
  --mask_validation \
  --grad_accumulation_steps=2 \
  --batch_size=8
```

Common parameters:

| Parameter | Purpose |
| --- | --- |
| `--version` | Directory or model name of the GLaMM initialization weights |
| `--vision_pretrained` | SAM ViT-H checkpoint |
| `--vision-tower` | Directory or model name of the CLIP vision tower |
| `--dataset_dir` | Root directory of the main training annotations; it must contain `reason_seg/ReasonSeg/train` |
| `--image_root_path` | Main training image directory |
| `--reason_seg_data_add` | Optional root directory of additional reasoning segmentation annotations |
| `--reason_image_root_path` | Directory of additional reasoning segmentation images |
| `--use_segm_data` | Enables training with segmentation data |
| `--seg_dataset="Refer_Segm"` | Uses PestSeg/ReasonSeg-style segmentation data |
| `--mask_validation` | Computes mask metrics during validation |
| `--pretrained` | Loads an existing grounding encoder/projection using the pretrained-model configuration |
| `--lora_r` | LoRA rank; set it to `0` to use the trainable non-LoRA branch |
| `--log_base_dir` / `--exp_name` | Output directory; by default, results are written to `output/<exp_name>/` |

Validation metrics are appended to the following file in the project root directory:

```text
results.txt
```

## Evaluation

The PestScope segmentation evaluation template is located at:

```bash
bash scripts/eval_pestscope_seg.sh
```

It also invokes `train_ft.py`, but additionally passes `--eval_only` and `--resume`:

```bash
deepspeed \
  --master_port=24969 \
  --include=localhost:0 \
  train_ft.py \
  --version="/path/to/GLaMM-RefSeg" \
  --dataset_dir="/path/to/refer_data/dif_data" \
  --reason_seg_data_add="/path/to/reason_data/dif_data" \
  --reason_image_root_path="/path/to/images" \
  --image_root_path="/path/to/images" \
  --vision_pretrained="/path/to/sam_vit_h_4b8939.pth" \
  --vision-tower="/path/to/clip-vit-large-patch14-336" \
  --exp_name="PestScope-7b" \
  --lora_r=8 \
  --pretrained \
  --use_segm_data \
  --seg_dataset="Refer_Segm" \
  --refer_segm_data="refcocog" \
  --val_dataset="RefCOCOgSegmVal" \
  --mask_validation \
  --batch_size=8 \
  --grad_accumulation_steps=2 \
  --eval_only \
  --resume="/path/to/output/PestScope-7b/ckpt_model_5/"
```

These results are also appended to `results.txt`.

If you are only evaluating the PDRES/referring segmentation data and do not need additional reasoning segmentation data, remove:

```text
--reason_seg_data_add
--reason_image_root_path
```

## Acknowledgements

We thank LISA, SAM, and GLaMM for releasing their models and code as open-source contributions.
