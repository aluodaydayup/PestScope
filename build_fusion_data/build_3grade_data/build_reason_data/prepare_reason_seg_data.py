from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


PestInfo = Dict[str, Dict[str, str]]


PEST24: PestInfo = {
    "1": {"英文名": "Rice planthopper", "中文名": "稻飞虱"},
    "2": {"英文名": "Rice Leaf Roller", "中文名": "稻纵卷叶螟"},
    "3": {"英文名": "Striped rice borer", "中文名": "二化螟"},
    "5": {"英文名": "Armyworm", "中文名": "黏虫"},
    "6": {"英文名": "Bollworm", "中文名": "棉铃虫"},
    "7": {"英文名": "Meadow borer", "中文名": "草地螟"},
    "8": {"英文名": "Athetis lepigone", "中文名": "二点委夜蛾"},
    "10": {"英文名": "Spodoptera litura", "中文名": "斜纹夜蛾"},
    "11": {"英文名": "Spodoptera exigua", "中文名": "甜菜夜蛾"},
    "12": {"英文名": "Stem borer", "中文名": "茎蛀虫"},
    "13": {"英文名": "Little Gecko", "中文名": "小地老虎"},
    "14": {"英文名": "Plutella xylostella", "中文名": "小菜蛾"},
    "15": {"英文名": "Spodoptera cabbage", "中文名": ""},
    "16": {"英文名": "Scotogramma trifolii Rottemberg", "中文名": "三叶草夜蛾"},
    "24": {"英文名": "Yellow tiger", "中文名": "黄地老虎"},
    "25": {"英文名": "Land tiger", "中文名": "小地老虎"},
    "28": {"英文名": "eight-character tiger", "中文名": "八字地老虎"},
    "29": {"英文名": "holotrichia oblita", "中文名": "大黑鳃金龟"},
    "31": {"英文名": "holotrichia parallela", "中文名": "暗黑鳃金龟"},
    "32": {"英文名": "Anomala corpulenta", "中文名": "铜绿丽金龟"},
    "34": {"英文名": "Gryllotalpa orientalis", "中文名": "东方蝼蛄"},
    "35": {"英文名": "Nematode trench", "中文名": "线虫"},
    "36": {"英文名": "Agriotes fuscicollis Miwa", "中文名": "金针虫"},
    "37": {"英文名": "Melahotus", "中文名": "麦蛾"},
}

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = SCRIPT_DIR / "source_data"
DEFAULT_KNOWLEDGE_JSON = DEFAULT_SOURCE_ROOT / "cleaned_data_add_num.json"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "generated_reason_data"
DEFAULT_SIMPLE_DIR = DEFAULT_OUTPUT_ROOT / "simple_data"
DEFAULT_MEDIUM_DIR = DEFAULT_OUTPUT_ROOT / "medium_data"
DEFAULT_HARD_DIR = DEFAULT_OUTPUT_ROOT / "dif_data"
QUESTION_PREFIXES = ["根据图片回答, ", "在图中, "]
DEFAULT_KNOWLEDGE_FIELDS = ["分布与危害", "形态特征", "发生规律"]


@dataclass(frozen=True)
class Preset:
    input_json: Path
    middle_json: Path
    reason_dir: Path
    summary_json: Path
    target_count_per_insect: int
    target_count_per_set: Sequence[int]
    target_weights: Sequence[int]
    seed: Optional[int]


PRESETS: Dict[str, Preset] = {
    "train": Preset(
        input_json=DEFAULT_MEDIUM_DIR / "medium_train_class_5.json",
        middle_json=DEFAULT_MEDIUM_DIR / "medium_new_train_class_5.json",
        reason_dir=DEFAULT_MEDIUM_DIR / "reason_seg" / "ReasonSeg" / "train",
        summary_json=DEFAULT_MEDIUM_DIR / "reason_seg" / "ReasonSeg" / "explanatory" / "train.json",
        target_count_per_insect=320,
        target_count_per_set=(1, 2, 3),
        target_weights=(2, 2, 1),
        seed=44,
    ),
    "val": Preset(
        input_json=DEFAULT_MEDIUM_DIR / "medium_val_class_5.json",
        middle_json=DEFAULT_MEDIUM_DIR / "medium_new_val_class_5.json",
        reason_dir=DEFAULT_MEDIUM_DIR / "reason_seg" / "ReasonSeg" / "val",
        summary_json=DEFAULT_MEDIUM_DIR / "reason_seg" / "ReasonSeg" / "explanatory" / "val.json",
        target_count_per_insect=320,
        target_count_per_set=(1, 2, 3),
        target_weights=(2, 2, 1),
        seed=44,
    ),
    "low-train": Preset(
        input_json=DEFAULT_SIMPLE_DIR / "simple_train_class_2.json",
        middle_json=DEFAULT_SIMPLE_DIR / "simple_new_train_class_2.json",
        reason_dir=DEFAULT_SIMPLE_DIR / "reason_seg" / "ReasonSeg" / "train",
        summary_json=DEFAULT_SIMPLE_DIR / "reason_seg" / "ReasonSeg" / "explanatory" / "train.json",
        target_count_per_insect=370,
        target_count_per_set=(1,),
        target_weights=(1,),
        seed=None,
    ),
    "low-val": Preset(
        input_json=DEFAULT_SIMPLE_DIR / "simple_val_class_2.json",
        middle_json=DEFAULT_SIMPLE_DIR / "simple_new_val_class_2.json",
        reason_dir=DEFAULT_SIMPLE_DIR / "reason_seg" / "ReasonSeg" / "val",
        summary_json=DEFAULT_SIMPLE_DIR / "reason_seg" / "ReasonSeg" / "explanatory" / "val.json",
        target_count_per_insect=370,
        target_count_per_set=(1,),
        target_weights=(1,),
        seed=None,
    ),
    "hard-train": Preset(
        input_json=DEFAULT_HARD_DIR / "hard_train_class_7.json",
        middle_json=DEFAULT_HARD_DIR / "hard_new_train_class_7.json",
        reason_dir=DEFAULT_HARD_DIR / "reason_seg" / "ReasonSeg" / "train",
        summary_json=DEFAULT_HARD_DIR / "reason_seg" / "ReasonSeg" / "explanatory" / "train.json",
        target_count_per_insect=320,
        target_count_per_set=(1, 2, 3),
        target_weights=(2, 2, 1),
        seed=44,
    ),
    "hard-val": Preset(
        input_json=DEFAULT_HARD_DIR / "hard_val_class_7.json",
        middle_json=DEFAULT_HARD_DIR / "hard_new_val_class_7.json",
        reason_dir=DEFAULT_HARD_DIR / "reason_seg" / "ReasonSeg" / "val",
        summary_json=DEFAULT_HARD_DIR / "reason_seg" / "ReasonSeg" / "explanatory" / "val.json",
        target_count_per_insect=320,
        target_count_per_set=(1, 2, 3),
        target_weights=(2, 2, 1),
        seed=44,
    ),
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def normalize_background_knowledge(raw_data: Any) -> Dict[str, Any]:
    if isinstance(raw_data, list):
        merged: Dict[str, Any] = {}
        for item in raw_data:
            if isinstance(item, dict):
                merged.update(item)
        return merged
    if isinstance(raw_data, dict):
        return raw_data
    raise TypeError("background knowledge json must be a dict or a list of dicts")


def all_pest_ids(data: Sequence[Mapping[str, Sequence[Mapping[str, Any]]]]) -> List[str]:
    ids = set()
    for row in data:
        for annotations in row.values():
            for annotation in annotations:
                ids.add(str(annotation["pest_id"]))
    return sorted(ids)


def select_target_annotations(
    input_path: Path,
    output_path: Optional[Path],
    target_count_per_insect: int,
    target_count_per_set: Sequence[int],
    target_weights: Sequence[int],
    seed: Optional[int] = None,
    save_middle: bool = True,
) -> List[Dict[str, List[Dict[str, Any]]]]:
    rng = random.Random(seed)
    source_data = load_json(input_path)
    source_data = copy.deepcopy(source_data)
    pest_ids = all_pest_ids(source_data)
    target_counter: Counter[str] = Counter()
    selected_data: List[Dict[str, List[Dict[str, Any]]]] = []

    for row in source_data:
        for image_id, annotations in row.items():
            if not annotations:
                selected_data.append({image_id: []})
                continue

            num_targets = rng.choices(target_count_per_set, weights=target_weights, k=1)[0]
            num_targets = min(num_targets, len(annotations))
            selected = rng.sample(list(annotations), num_targets)
            selected_ids = {id(item) for item in selected}
            unselected = [item for item in annotations if id(item) not in selected_ids]

            for annotation in unselected:
                annotation["is_target"] = False
            for annotation in selected:
                pest_id = str(annotation["pest_id"])
                target_counter[pest_id] += 1
                annotation["is_target"] = True

            selected_data.append({image_id: unselected + selected})

        if target_count_per_insect > 0 and pest_ids:
            if all(target_counter[pest_id] >= target_count_per_insect for pest_id in pest_ids):
                break

    if save_middle and output_path is not None:
        write_json(output_path, selected_data)
    print("target_insect_counter", target_counter)
    return selected_data


def annotation_to_reason_item(annotation: Mapping[str, Any], pest_info: PestInfo) -> Dict[str, Any]:
    pest_id = str(annotation["pest_id"])
    info = pest_info.get(pest_id, {})
    bbox_list = []
    seg_list = []
    for pest in annotation.get("annotation", []):
        bbox_list.append(pest["bbox"])
        seg_list.append(pest["mask_path"])

    return {
        "pest_id": pest_id,
        "en_name": info.get("英文名", ""),
        "pest_name": info.get("中文名", ""),
        "bbox": bbox_list,
        "segmentation": seg_list,
        "is_target": bool(annotation.get("is_target", False)),
    }


def choose_question_snippet(
    pest_id: str,
    pest_name: str,
    background_knowledge: Mapping[str, Any],
    rng: random.Random,
    knowledge_fields: Sequence[str],
) -> str:
    knowledge_item = background_knowledge.get(str(pest_id), {})
    if not isinstance(knowledge_item, Mapping):
        return ""

    background = knowledge_item.get("背景知识", {})
    if not isinstance(background, Mapping):
        return ""

    candidate_fields = [field for field in knowledge_fields if field in background and background[field]]
    if not candidate_fields:
        candidate_fields = [field for field, values in background.items() if values]
    if not candidate_fields:
        return ""

    field = rng.choice(candidate_fields)
    sentence = rng.choice(list(background[field]))
    return str(sentence).replace(pest_name, "哪种害虫")


def build_reason_json(
    image_id: str,
    annotations: Sequence[Mapping[str, Any]],
    pest_info: PestInfo,
    background_knowledge: Mapping[str, Any],
    rng: random.Random,
    knowledge_fields: Sequence[str],
) -> Dict[str, Any]:
    reason_annotations = [annotation_to_reason_item(annotation, pest_info) for annotation in annotations]
    question = rng.choice(QUESTION_PREFIXES)
    for annotation in reason_annotations:
        if annotation["is_target"]:
            question += choose_question_snippet(
                pest_id=annotation["pest_id"],
                pest_name=annotation["pest_name"],
                background_knowledge=background_knowledge,
                rng=rng,
                knowledge_fields=knowledge_fields,
            )

    return {
        "text": [question],
        "is_sentence": True,
        "shapes": [
            {
                "label": "target",
                "labels": ["target"],
                "shape_type": "polygon",
                "image_name": f"{image_id}.jpg",
                "points": [],
                "group_id": None,
                "group_ids": [None],
                "flags": {},
            }
        ],
        "ann": reason_annotations,
    }


def write_reason_files(
    selected_data: Sequence[Mapping[str, Sequence[Mapping[str, Any]]]],
    output_dir: Path,
    pest_info: PestInfo,
    background_knowledge: Mapping[str, Any],
    seed: Optional[int] = None,
    knowledge_fields: Sequence[str] = DEFAULT_KNOWLEDGE_FIELDS,
) -> List[Path]:
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: List[Path] = []
    for row in selected_data:
        for image_id, annotations in row.items():
            reason_json = build_reason_json(
                image_id=str(image_id),
                annotations=annotations,
                pest_info=pest_info,
                background_knowledge=background_knowledge,
                rng=rng,
                knowledge_fields=knowledge_fields,
            )
            output_path = output_dir / f"{image_id}.json"
            write_json(output_path, reason_json)
            written_paths.append(output_path)
    return written_paths


def get_query(text: Any) -> str:
    if isinstance(text, list):
        return str(text[0]) if text else ""
    return str(text)


def write_explanatory_json(reason_dir: Path, output_path: Path) -> List[Dict[str, str]]:
    all_data: List[Dict[str, str]] = []
    for json_path in sorted(reason_dir.glob("*.json")):
        reason_data = load_json(json_path)
        output_list = [
            item.get("pest_name", "")
            for item in reason_data.get("ann", [])
            if item.get("is_target", False)
        ]
        all_data.append(
            {
                "query": get_query(reason_data.get("text", "")),
                "image": f"{json_path.stem}.jpg",
                "json": json_path.name,
                "outputs": ",".join(output_list) if output_list else " ",
            }
        )
    write_json(output_path, all_data)
    return all_data


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    preset = PRESETS[args.preset]
    return {
        "input_json": args.input_json or preset.input_json,
        "middle_json": args.middle_json or preset.middle_json,
        "reason_dir": args.reason_dir or preset.reason_dir,
        "summary_json": args.summary_json or preset.summary_json,
        "target_count_per_insect": args.target_count_per_insect or preset.target_count_per_insect,
        "target_count_per_set": parse_int_list(args.target_count_per_set) if args.target_count_per_set else list(preset.target_count_per_set),
        "target_weights": parse_int_list(args.target_weights) if args.target_weights else list(preset.target_weights),
        "seed": preset.seed if args.seed is None else args.seed,
    }


def run(args: argparse.Namespace) -> None:
    config = resolve_config(args)
    selected_data = None

    if args.step in {"select-targets", "all"}:
        selected_data = select_target_annotations(
            input_path=config["input_json"],
            output_path=config["middle_json"],
            target_count_per_insect=config["target_count_per_insect"],
            target_count_per_set=config["target_count_per_set"],
            target_weights=config["target_weights"],
            seed=config["seed"],
            save_middle=not args.no_save_middle,
        )

    if args.step in {"reason-files", "all"}:
        if selected_data is None:
            selected_data = load_json(config["middle_json"])
        knowledge_path = args.knowledge_json
        background_knowledge = normalize_background_knowledge(load_json(knowledge_path))
        written_paths = write_reason_files(
            selected_data=selected_data,
            output_dir=config["reason_dir"],
            pest_info=PEST24,
            background_knowledge=background_knowledge,
            seed=config["seed"],
            knowledge_fields=args.knowledge_fields,
        )
        print(f"wrote_reason_files {len(written_paths)}")

    if args.step in {"summary", "all"}:
        summary = write_explanatory_json(config["reason_dir"], config["summary_json"])
        print(f"wrote_summary_items {len(summary)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare reason segmentation data for train/val and low difficulty val.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="train", help="Default path and sampling configuration.")
    parser.add_argument("--step", choices=["select-targets", "reason-files", "summary", "all"], default="all")
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--middle-json", type=Path, default=None)
    parser.add_argument("--reason-dir", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--knowledge-json", type=Path, default=DEFAULT_KNOWLEDGE_JSON)
    parser.add_argument("--target-count-per-insect", type=int, default=None)
    parser.add_argument("--target-count-per-set", default=None, help="Comma-separated target counts, for example: 1,2,3")
    parser.add_argument("--target-weights", default=None, help="Comma-separated weights, for example: 2,2,1")
    parser.add_argument("--knowledge-fields", nargs="*", default=DEFAULT_KNOWLEDGE_FIELDS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-save-middle", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
