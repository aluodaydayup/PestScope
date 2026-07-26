from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from tqdm import tqdm


PEST_IDS = [
    "1",
    "2",
    "3",
    "5",
    "6",
    "7",
    "8",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "24",
    "25",
    "28",
    "29",
    "31",
    "32",
    "34",
    "35",
    "36",
    "37",
]

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = SCRIPT_DIR / "source_data"

DEFAULT_DATABASE_PATH = str(DEFAULT_SOURCE_ROOT / "pest24_database.json")
DEFAULT_BACKGROUND_DIR = str(DEFAULT_SOURCE_ROOT / "background_images")
DEFAULT_MASK_ROOT = str(DEFAULT_SOURCE_ROOT / "mask_image")
DEFAULT_IMAGE_ROOT = str(DEFAULT_SOURCE_ROOT / "JPEGImages")
DEFAULT_OUTPUT_ROOT = str(SCRIPT_DIR / "generated_refer_data")


@dataclass(frozen=True)
class DifficultyConfig:
    name: str
    output_subdir: str
    filename_prefix: str
    class_num: int
    extra_pest_count: int
    min_extra_unique_count: int
    train_outer_loops: int
    val_outer_loops: int
    excluded_pest_ids: Tuple[str, ...]


DIFFICULTY_CONFIGS: Dict[str, DifficultyConfig] = {
    "simple": DifficultyConfig(
        name="simple",
        output_subdir="simple_data",
        filename_prefix="refer",
        class_num=2,
        extra_pest_count=0,
        min_extra_unique_count=0,
        train_outer_loops=720,
        val_outer_loops=200,
        excluded_pest_ids=(
            "1",
            "2",
            "3",
            "4",
            "5",
            "7",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "24",
            "25",
            "28",
            "29",
            "31",
            "35",
            "37",
        ),
    ),
    "medium": DifficultyConfig(
        name="medium",
        output_subdir="medium_data",
        filename_prefix="refer",
        class_num=5,
        extra_pest_count=3,
        min_extra_unique_count=0,
        train_outer_loops=109,
        val_outer_loops=30,
        excluded_pest_ids=("1", "12", "13", "14", "15", "31", "36", "37", "5", "24", "25", "29"),
    ),
    "hard": DifficultyConfig(
        name="hard",
        output_subdir="dif_data",
        filename_prefix="refer",
        class_num=7,
        extra_pest_count=5,
        min_extra_unique_count=2,
        train_outer_loops=47,
        val_outer_loops=13,
        excluded_pest_ids=("1", "12", "13", "14", "15", "37"),
    ),
}

DIFFICULTY_ALIASES = {
    "low": "simple",
    "simple": "simple",
    "medium": "medium",
    "hard": "hard",
    "dif": "hard",
}


def apply_shadow(image, mask, shadow_intensity: float = 0.5):
    import cv2
    import numpy as np

    shadow = np.zeros_like(image)
    shadow_part = cv2.bitwise_and(shadow, shadow, mask=mask)
    return cv2.addWeighted(shadow_part, shadow_intensity, shadow_part, 0, 0)


def compose_image(
    original_image_paths: Sequence[str],
    mask_image_paths: Sequence[str],
    background_image_path: str,
    output_image_path: str,
):
    import cv2
    import numpy as np

    new_background = cv2.imread(background_image_path)
    if new_background is None:
        raise FileNotFoundError(f"Background image not found or unreadable: {background_image_path}")

    if original_image_paths:
        original_image = cv2.imread(original_image_paths[0])
        if original_image is None:
            raise FileNotFoundError(f"Original image not found or unreadable: {original_image_paths[0]}")
        new_background = cv2.resize(new_background, (original_image.shape[1], original_image.shape[0]))

    for original_path, mask_path in zip(original_image_paths, mask_image_paths):
        original_image = cv2.imread(original_path)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if original_image is None:
            raise FileNotFoundError(f"Original image not found or unreadable: {original_path}")
        if mask_image is None:
            raise FileNotFoundError(f"Mask image not found or unreadable: {mask_path}")

        composite_mask = np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.uint8)
        composite_mask = cv2.bitwise_or(composite_mask, mask_image)
        masked_part = cv2.bitwise_and(original_image, original_image, mask=composite_mask)
        shadow_part = apply_shadow(original_image, composite_mask)
        inverse_mask = cv2.bitwise_not(composite_mask)
        background_cleared = cv2.bitwise_and(new_background, new_background, mask=inverse_mask)
        new_background = cv2.add(background_cleared, masked_part)
        new_background = cv2.add(new_background, shadow_part)

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, new_background)
    return new_background


def has_overlap(boxes: Sequence[Sequence[int]]) -> bool:
    def is_overlap(box1: Sequence[int], box2: Sequence[int]) -> bool:
        x_min1, y_min1, x_max1, y_max1 = box1
        x_min2, y_min2, x_max2, y_max2 = box2
        intersection_width = max(0, min(x_max1, x_max2) - max(x_min1, x_min2))
        intersection_height = max(0, min(y_max1, y_max2) - max(y_min1, y_min2))
        return intersection_width > 0 and intersection_height > 0

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if is_overlap(boxes[i], boxes[j]):
                return True
    return False


def is_within_bounds(x_min: int, x_max: int, y_min: int, y_max: int) -> bool:
    return 50 <= x_min <= 750 and 50 <= x_max <= 750 and 50 <= y_min <= 550 and 50 <= y_max <= 550


def load_pest_database(database_path: str) -> List[dict]:
    with open(database_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_background_images(background_dir: str) -> List[str]:
    background_images = []
    for root, _, files in os.walk(background_dir):
        for file in files:
            background_images.append(os.path.join(root, file))
    if not background_images:
        raise FileNotFoundError(f"No background images found in: {background_dir}")
    return background_images


def get_available_pest_ids(config: DifficultyConfig) -> List[str]:
    excluded = set(config.excluded_pest_ids)
    return [pest_id for pest_id in PEST_IDS if pest_id not in excluded]


def choose_extra_pest_ids(
    other_ids: Sequence[str],
    extra_pest_count: int,
    min_extra_unique_count: int,
) -> List[str]:
    if extra_pest_count == 0:
        return []
    if min_extra_unique_count > extra_pest_count:
        raise ValueError("min_extra_unique_count cannot exceed extra_pest_count")
    if min_extra_unique_count > len(other_ids):
        raise ValueError("min_extra_unique_count cannot exceed available extra pest ids")

    required_unique_ids = random.sample(list(other_ids), k=min_extra_unique_count)
    remaining_count = extra_pest_count - min_extra_unique_count
    weights = [1 / len(other_ids)] * len(other_ids)
    remaining_ids = random.choices(other_ids, weights=weights, k=remaining_count)
    return required_unique_ids + remaining_ids


def build_combinations(
    pest_ids: Sequence[str],
    extra_pest_count: int,
    min_extra_unique_count: int = 0,
) -> List[Tuple[str, ...]]:
    combinations = []
    for i in range(len(pest_ids)):
        for j in range(i + 1, len(pest_ids)):
            base_ids = [pest_ids[i], pest_ids[j]]
            other_ids = [pest_id for pest_id in pest_ids if pest_id not in base_ids]
            extra_ids = choose_extra_pest_ids(other_ids, extra_pest_count, min_extra_unique_count)
            combinations.append(tuple(base_ids + extra_ids))
    return combinations


def select_pest_annotation(pest_database: List[dict], pest_id: str, max_attempts: int = 1000) -> dict:
    last_pest = None
    for _ in range(max_attempts):
        pest = random.choice(pest_database[0][pest_id]["in_file"])
        last_pest = pest
        x_min, y_min, x_max, y_max = pest["bbox"]
        if is_within_bounds(x_min, x_max, y_min, y_max):
            return pest
    if last_pest is not None:
        return last_pest
    raise ValueError(f"No annotation candidates found for pest id: {pest_id}")


def output_stem(config: DifficultyConfig, data_type: str, entity_num: int, num: int) -> str:
    return f"{config.name}_{config.filename_prefix}_{data_type}_class_{config.class_num}_{entity_num}_{num}"


def default_outer_loops(config: DifficultyConfig, data_type: str) -> int:
    return config.train_outer_loops if data_type == "train" else config.val_outer_loops


def process_combination(
    combination: Sequence[str],
    num: int,
    data_type: str,
    config: DifficultyConfig,
    pest_database: List[dict],
    background_images: Sequence[str],
    output_root: str,
    mask_root: str,
    image_root: str,
) -> dict:
    overlap = True
    while overlap:
        pest_list = []
        pest_dict = {}
        for pest_id in combination:
            pest = select_pest_annotation(pest_database, pest_id)
            pest_list.append(pest)
            pest_dict.setdefault(pest_id, []).append(pest)

        bbox_list = [pest["bbox"] for pest in pest_list]
        overlap = has_overlap(bbox_list)

    original_image_paths = []
    mask_image_paths = []
    for values in pest_dict.values():
        for value in values:
            filename = str(value["file_name"])
            original_image_paths.append(os.path.join(image_root, f"{filename}.jpg"))
            mask_image_paths.append(os.path.join(mask_root, value["mask_path"].split("/")[-1]))

    entity_num = len(mask_image_paths)
    file_stem = output_stem(config, data_type, entity_num, num)
    output_image_path = os.path.join(output_root, config.output_subdir, "images", f"{file_stem}.jpg")
    compose_image(original_image_paths, mask_image_paths, random.choice(background_images), output_image_path)

    annotations = []
    for pest_id, values in pest_dict.items():
        annotations.append(
            {
                "pest_id": pest_id,
                "quantity": len(values),
                "annotation": values,
            }
        )
    return {file_stem: annotations}


def run_generation(
    difficulty: str,
    data_type: str = "val",
    database_path: str = DEFAULT_DATABASE_PATH,
    background_dir: str = DEFAULT_BACKGROUND_DIR,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    mask_root: str = DEFAULT_MASK_ROOT,
    image_root: str = DEFAULT_IMAGE_ROOT,
    outer_loops: int = None,
    workers: int = None,
    seed: int = None,
) -> List[dict]:
    config_key = DIFFICULTY_ALIASES.get(difficulty, difficulty)
    if config_key not in DIFFICULTY_CONFIGS:
        raise ValueError(f"Unknown difficulty: {difficulty}. Choose from: {', '.join(DIFFICULTY_CONFIGS)}")
    config = DIFFICULTY_CONFIGS[config_key]

    if seed is not None:
        random.seed(seed)

    pest_database = load_pest_database(database_path)
    background_images = list_background_images(background_dir)
    pest_ids = get_available_pest_ids(config)
    combinations = build_combinations(pest_ids, config.extra_pest_count, config.min_extra_unique_count)

    num = 0
    annotations = []
    loop_count = outer_loops if outer_loops is not None else default_outer_loops(config, data_type)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for _ in tqdm(range(0, loop_count), desc="Processing outer loop"):
            for combination in combinations:
                futures.append(
                    executor.submit(
                        process_combination,
                        combination,
                        num,
                        data_type,
                        config,
                        pest_database,
                        background_images,
                        output_root,
                        mask_root,
                        image_root,
                    )
                )
                num += 1

        for future in concurrent.futures.as_completed(futures):
            annotations.append(future.result())

    output_json_path = os.path.join(
        output_root,
        config.output_subdir,
        f"{config.name}_{data_type}_class_{config.class_num}.json",
    )
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=4)
    return annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate refer data for simple, medium, or hard difficulty.")
    parser.add_argument("--difficulty", choices=sorted(DIFFICULTY_CONFIGS), required=True)
    parser.add_argument("--data-type", choices=["train", "val"], default="val")
    parser.add_argument("--database-path", default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--background-dir", default=DEFAULT_BACKGROUND_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--mask-root", default=DEFAULT_MASK_ROOT)
    parser.add_argument("--image-root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--outer-loops", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_generation(
        difficulty=args.difficulty,
        data_type=args.data_type,
        database_path=args.database_path,
        background_dir=args.background_dir,
        output_root=args.output_root,
        mask_root=args.mask_root,
        image_root=args.image_root,
        outer_loops=args.outer_loops,
        workers=args.workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
