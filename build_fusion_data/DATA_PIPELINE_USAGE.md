## 1. Overall Workflow

```text
Pest24 raw images + XML bbox + pest knowledge text
    -> Generate instance masks and write mask_path back to the XML files
    -> Build pest24_database.json from XML files that contain mask_path
    -> Generate, clean, and unify pest background knowledge files
    -> Synthesize fused images for the simple / medium / hard difficulty levels
    -> Generate intermediate annotation JSON files for reason or refer data
    -> Convert them into per-image ReasonSeg / ReferSeg JSON files
    -> Generate explanatory/train.json and explanatory/val.json
```

If the following files and directories already exist, you can skip `data_process` and start directly from step 5:

```text
pest24_database.json
JPEGImages/
mask_image/
background_images/
cleaned_data_add_num.json    # Required only for reason data
```

## 2. Directory Responsibilities

### data_process

`data_process` is the upstream preprocessing directory. It builds the basic data required by the synthesis stage from the raw Pest24 data.

| File | Purpose |
| --- | --- |
| `get_pest_mask.py` | Uses SAM2 to generate pest instance masks from bbox annotations in XML files, and writes `mask_path` back to the XML files. |
| `build_pest24_database.py` | Iterates through XML files that contain `mask_path` and generates `pest24_database.json`. |
| `TopicSentenceForPest.py` | Calls an OpenAI-compatible API to split pest knowledge text into topic sentences. |
| `CleanDirtyCharacters.py` | Cleans model output by removing markdown wrappers, line breaks, and tabs, and converts stringified lists into real lists. |
| `PestDataUnity.py` | Organizes cleaned pest knowledge and Chinese/English name mappings into a unified format. |
| `test_pest_usage.md` | Usage instructions for the SAM2 mask generation script. |

### build_3grade_data

`build_3grade_data` is the downstream data synthesis directory. It builds training data by task type and difficulty.

```text
build_3grade_data/
  build_reason_data/    # Reasoning segmentation data
  build_refer_data/     # Referring expression / reference segmentation data
```

## 3. Prepare Input Data

The recommended directory structure is shown below. Specific paths can be overridden with command-line arguments.

```text
build_fusion_data/
  pest24_database.json
  build_3grade_data/
    build_reason_data/
      source_data/
        pest24_database.json
        background_images/
        mask_image/
        JPEGImages/
        cleaned_data_add_num.json
    build_refer_data/
      source_data/
        pest24_database.json
        background_images/
        mask_image/
        JPEGImages/
```

## 4. Upstream Preprocessing Workflow

### 4.1 Generate Masks

Script:

```text
build_fusion_data/data_process/get_pest_mask.py
```

Default inputs:

```text
Models/sam2_checkpoint/sam2_hiera_large.pt
pest24_data/Pest24/JPEGImages/
pest24_data/Pest24/Annotations/
```

Output:

```text
pest24_data/Pest24/mask_image/*.png
```

The script also modifies the XML files by adding the following field to matched `object` nodes:

```xml
<mask_path>Pest24/mask_image/xxx.png</mask_path>
```

Run example:

```bash
python build_fusion_data/data_process/get_pest_mask.py
```

Note: this script depends on `torch`, `sam2`, `numpy`, `Pillow`, and `opencv-python`, and uses `CUDA_VISIBLE_DEVICES=0` by default.

### 4.2 Build pest24_database.json

Script:

```text
build_fusion_data/data_process/build_pest24_database.py
```

Run example:

```bash
python build_fusion_data/data_process/build_pest24_database.py \
  --annotation-dir pest24_data/Pest24/Annotations \
  --output build_fusion_data/pest24_database.json
```

### 4.3 Generate and Clean Pest Background Knowledge

Related scripts:

```text
TopicSentenceForPest.py
CleanDirtyCharacters.py
PestDataUnity.py
```

Recommended order:

```text
TopicSentenceForPest.py
    -> pest_topic_sentence.json
CleanDirtyCharacters.py
    -> pest_topic_sentence_clean.json
PestDataUnity.py
    -> pest_unity.json or cleaned_data_add_num.json
```

The reason data preparation stage reads the following file by default:

```text
build_3grade_data/build_reason_data/source_data/cleaned_data_add_num.json
```

## 5. Build ReasonSeg Data

Directory:

```text
build_fusion_data/build_3grade_data/build_reason_data
```

### 5.1 Generate Fused Images and Intermediate Annotations

Script:

```text
generate_reason_data.py
```

Default inputs:

```text
source_data/pest24_database.json
source_data/background_images/
source_data/mask_image/
source_data/JPEGImages/
```

Output:

```text
generated_reason_data/<difficulty_dir>/images/*.jpg
generated_reason_data/<difficulty_dir>/<difficulty>_<train|val>_class_<N>.json
```

Run examples:

```bash
cd build_fusion_data/build_3grade_data/build_reason_data

python generate_reason_data.py --difficulty simple --data-type train
python generate_reason_data.py --difficulty simple --data-type val

python generate_reason_data.py --difficulty medium --data-type train
python generate_reason_data.py --difficulty medium --data-type val

python generate_reason_data.py --difficulty hard --data-type train
python generate_reason_data.py --difficulty hard --data-type val
```

### 5.2 Generate Per-Image ReasonSeg JSON

Script:

```text
prepare_reason_seg_data.py
```

Run examples:

```bash
cd build_fusion_data/build_3grade_data/build_reason_data

python prepare_reason_seg_data.py --preset low-train --step all
python prepare_reason_seg_data.py --preset low-val --step all

python prepare_reason_seg_data.py --preset train --step all
python prepare_reason_seg_data.py --preset val --step all

python prepare_reason_seg_data.py --preset hard-train --step all
python prepare_reason_seg_data.py --preset hard-val --step all
```

Output:

```text
generated_reason_data/<difficulty_dir>/reason_seg/ReasonSeg/train/*.json
generated_reason_data/<difficulty_dir>/reason_seg/ReasonSeg/val/*.json
generated_reason_data/<difficulty_dir>/reason_seg/ReasonSeg/explanatory/train.json
generated_reason_data/<difficulty_dir>/reason_seg/ReasonSeg/explanatory/val.json
```

## 6. Build ReferSeg Data

Directory:

```text
build_fusion_data/build_3grade_data/build_refer_data
```

### 6.1 Generate Fused Images and Intermediate Annotations

Script:

```text
generate_refer_data.py
```

Run examples:

```bash
cd build_fusion_data/build_3grade_data/build_refer_data

python generate_refer_data.py --difficulty simple --data-type train
python generate_refer_data.py --difficulty simple --data-type val

python generate_refer_data.py --difficulty medium --data-type train
python generate_refer_data.py --difficulty medium --data-type val

python generate_refer_data.py --difficulty hard --data-type train
python generate_refer_data.py --difficulty hard --data-type val
```

Output:

```text
generated_refer_data/<difficulty_dir>/images/*.jpg
generated_refer_data/<difficulty_dir>/<difficulty>_<train|val>_class_<N>.json
```

### 6.2 Convert to Per-Image ReferSeg JSON

Script:

```text
prepare_refer_seg_data.py
```

Run examples:

```bash
cd build_fusion_data/build_3grade_data/build_refer_data

python prepare_refer_seg_data.py --preset low-train --step all
python prepare_refer_seg_data.py --preset low-val --step all

python prepare_refer_seg_data.py --preset train --step all
python prepare_refer_seg_data.py --preset val --step all

python prepare_refer_seg_data.py --preset hard-train --step all
python prepare_refer_seg_data.py --preset hard-val --step all
```

Output:

```text
generated_refer_data/<difficulty_dir>/reason_seg/ReasonSeg/train/*.json
generated_refer_data/<difficulty_dir>/reason_seg/ReasonSeg/val/*.json
generated_refer_data/<difficulty_dir>/reason_seg/ReasonSeg/explanatory/train.json
generated_refer_data/<difficulty_dir>/reason_seg/ReasonSeg/explanatory/val.json
```

The directory name here is still `reason_seg/ReasonSeg`, but the data content follows the refer task format: `text` is an empty string and `is_sentence` is `False`.

## 7. Complete Command Examples

### Build only medium reason train/val

```bash
cd build_fusion_data/build_3grade_data/build_reason_data

python generate_reason_data.py --difficulty medium --data-type train
python prepare_reason_seg_data.py --preset train --step all

python generate_reason_data.py --difficulty medium --data-type val
python prepare_reason_seg_data.py --preset val --step all
```

### Build only medium refer train/val

```bash
cd build_fusion_data/build_3grade_data/build_refer_data

python generate_refer_data.py --difficulty medium --data-type train
python prepare_refer_seg_data.py --preset train --step all

python generate_refer_data.py --difficulty medium --data-type val
python prepare_refer_seg_data.py --preset val --step all
```
