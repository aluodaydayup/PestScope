from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


PestInfo = Dict[str, Dict[str, str]]


PEST24: PestInfo = {
    "1": {"英文名": "Rice planthopper", "中文名": "稻飞虱"},
    "2": {"英文名": "Rice Leaf Roller", "中文名": "稻纵卷叶螟"},
    "3": {"英文名": "Striped rice borer", "中文名": "二化螟"},
    "5": {"英文名": "Armyworm", "中文名": "粘虫"},
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
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "generated_refer_data"
DEFAULT_SIMPLE_DIR = DEFAULT_OUTPUT_ROOT / "simple_data"
DEFAULT_MEDIUM_DIR = DEFAULT_OUTPUT_ROOT / "medium_data"
DEFAULT_HARD_DIR = DEFAULT_OUTPUT_ROOT / "dif_data"


@dataclass(frozen=True)
class Preset:
    input_json: Path
    reason_dir: Path
    summary_json: Path


PRESETS: Dict[str, Preset] = {
    "train": Preset(
        input_json=DEFAULT_MEDIUM_DIR / "medium_train_class_5.json",
        reason_dir=DEFAULT_MEDIUM_DIR / "reason_seg" / "ReasonSeg" / "train",
        summary_json=DEFAULT_MEDIUM_DIR / "reason_seg" / "ReasonSeg" / "explanatory" / "train.json",
    ),
    "val": Preset(
        input_json=DEFAULT_MEDIUM_DIR / "medium_val_class_5.json",
        reason_dir=DEFAULT_MEDIUM_DIR / "reason_seg" / "ReasonSeg" / "val",
        summary_json=DEFAULT_MEDIUM_DIR / "reason_seg" / "ReasonSeg" / "explanatory" / "val.json",
    ),
    "low-train": Preset(
        input_json=DEFAULT_SIMPLE_DIR / "simple_train_class_2.json",
        reason_dir=DEFAULT_SIMPLE_DIR / "reason_seg" / "ReasonSeg" / "train",
        summary_json=DEFAULT_SIMPLE_DIR / "reason_seg" / "ReasonSeg" / "explanatory" / "train.json",
    ),
    "low-val": Preset(
        input_json=DEFAULT_SIMPLE_DIR / "simple_val_class_2.json",
        reason_dir=DEFAULT_SIMPLE_DIR / "reason_seg" / "ReasonSeg" / "val",
        summary_json=DEFAULT_SIMPLE_DIR / "reason_seg" / "ReasonSeg" / "explanatory" / "val.json",
    ),
    "hard-train": Preset(
        input_json=DEFAULT_HARD_DIR / "hard_train_class_7.json",
        reason_dir=DEFAULT_HARD_DIR / "reason_seg" / "ReasonSeg" / "train",
        summary_json=DEFAULT_HARD_DIR / "reason_seg" / "ReasonSeg" / "explanatory" / "train.json",
    ),
    "hard-val": Preset(
        input_json=DEFAULT_HARD_DIR / "hard_val_class_7.json",
        reason_dir=DEFAULT_HARD_DIR / "reason_seg" / "ReasonSeg" / "val",
        summary_json=DEFAULT_HARD_DIR / "reason_seg" / "ReasonSeg" / "explanatory" / "val.json",
    ),
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def annotation_to_refer_item(annotation: Mapping[str, Any], pest_info: PestInfo) -> Dict[str, Any]:
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
    }


def build_refer_json(
    image_id: str,
    annotations: Sequence[Mapping[str, Any]],
    pest_info: PestInfo,
) -> Dict[str, Any]:
    return {
        "text": "",
        "is_sentence": False,
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
        "ann": [annotation_to_refer_item(annotation, pest_info) for annotation in annotations],
    }


def write_refer_files(
    source_data: Sequence[Mapping[str, Sequence[Mapping[str, Any]]]],
    output_dir: Path,
    pest_info: PestInfo,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: List[Path] = []
    for row in source_data:
        for image_id, annotations in row.items():
            refer_json = build_refer_json(
                image_id=str(image_id),
                annotations=annotations,
                pest_info=pest_info,
            )
            output_path = output_dir / f"{image_id}.json"
            write_json(output_path, refer_json)
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


def resolve_config(args: argparse.Namespace) -> Dict[str, Path]:
    preset = PRESETS[args.preset]
    return {
        "input_json": args.input_json or preset.input_json,
        "reason_dir": args.reason_dir or preset.reason_dir,
        "summary_json": args.summary_json or preset.summary_json,
    }


def run(args: argparse.Namespace) -> None:
    config = resolve_config(args)

    if args.step in {"reason-files", "all"}:
        source_data = load_json(config["input_json"])
        written_paths = write_refer_files(
            source_data=source_data,
            output_dir=config["reason_dir"],
            pest_info=PEST24,
        )
        print(f"wrote_refer_files {len(written_paths)}")

    if args.step in {"summary", "all"}:
        summary = write_explanatory_json(config["reason_dir"], config["summary_json"])
        print(f"wrote_summary_items {len(summary)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare refer segmentation data for train/val splits.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="train", help="Default path configuration.")
    parser.add_argument("--step", choices=["reason-files", "summary", "all"], default="all")
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--reason-dir", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
