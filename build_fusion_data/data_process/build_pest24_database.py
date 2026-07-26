import argparse
import copy
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

pest24 = {"1": {"英文名": "Rice planthopper", "中文名": "稻飞虱", "知识路径": "稻飞虱.txt"},
"2": {"英文名": "Rice Leaf Roller", "中文名": "稻纵卷叶螟",
"知识路径": "稻纵卷叶螟.txt"},
"3": {"英文名": "Striped rice borer", "中文名": "二化螟",
"知识路径": "二化螟.txt"},
"5": {"英文名": "Armyworm", "中文名": "黏虫", "知识路径": "黏虫.txt"},
"6": {"英文名": "Bollworm", "中文名": "棉铃虫", "知识路径": "棉铃虫.txt"},
"7": {"英文名": "Meadow borer", "中文名": "草地螟", "知识路径": "草地螟.txt"},
"8": {"英文名": "Athetis lepigone", "中文名": "二点委夜蛾",
"知识路径": "二点委夜蛾.txt"},
"10": {"英文名": "Spodoptera litura", "中文名": "斜纹夜蛾",
"知识路径": "斜纹夜蛾.txt"},
"11": {"英文名": "Spodoptera exigua", "中文名": "甜菜夜蛾",
"知识路径": "甜菜夜蛾.txt"},
"12": {"英文名": "Stem borer", "中文名": "茎蛀虫?", "知识路径": ""},
"13": {"英文名": "Little Gecko", "中文名": "小壁虎?", "知识路径": ""},
"14": {"英文名": "Plutella xylostella", "中文名": "小菜蛾",
"知识路径": "小菜蛾.txt"},
"15": {"英文名": "Spodoptera cabbage", "中文名": "", "知识路径": ""},
"16": {"英文名": "Scotogramma trifolii Rottemberg", "中文名": "三叶草夜蛾",
"知识路径": "三叶草夜蛾.txt"},
"24": {"英文名": "Yellow tiger", "中文名": "黄地老虎",
"知识路径": "黄地老虎.txt"},
"25": {"英文名": "Land tiger", "中文名": "小地老虎", "知识路径": "小地老虎.txt"},
"28": {"英文名": "eight-character tiger", "中文名": "八字地老虎",
"知识路径": "八字地老虎.txt"},
"29": {"英文名": "holotrichia oblita", "中文名": "大黑鳃金龟",
"知识路径": "大黑鳃金龟.txt"},
"31": {"英文名": "holotrichia parallela", "中文名": "暗黑鳃金龟",
"知识路径": "暗黑鳃金龟.txt"},
"32": {"英文名": "Anomala corpulenta", "中文名": "铜绿丽金龟",
"知识路径": "铜绿丽金龟.txt"},
"34": {"英文名": "Gryllotalpa orientalis", "中文名": "东方蝼蛄",
"知识路径": "东方蝼蛄.txt"},
"35": {"英文名": "Nematode trench", "中文名": "线虫", "知识路径": "线虫.txt"},
"36": {"英文名": "Agriotes fuscicollis Miwa", "中文名": "金针虫",
"知识路径": "金针虫.txt"},
"37": {"英文名": "Melahotus", "中文名": "麦蛾", "知识路径": "麦蛾.txt"}
}

DEFAULT_ANNOTATION_DIR = Path("Annotations_add_mask_clean")
DEFAULT_OUTPUT_PATH = Path("pest24_database.json")


def _load_template(template_path):
    with Path(template_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise ValueError("template JSON must be a list containing one database object")
    return data[0]


def _empty_database_from_template(template_database):
    database = copy.deepcopy(template_database)
    for pest_info in database.values():
        pest_info["in_file"] = []
    return database


def _empty_database_from_pest24():
    database = {}
    for pest_id, pest_info in pest24.items():
        database[pest_id] = {
            "英文名": pest_info.get("英文名", ""),
            "中文名": pest_info.get("中文名", ""),
            "in_file": [],
        }
    return database


def _initial_database(template_path=None):
    if template_path:
        return _empty_database_from_template(_load_template(template_path))
    return _empty_database_from_pest24()


def _find_xml_files(annotation_dir):
    annotation_dir = Path(annotation_dir)
    if not annotation_dir.exists():
        raise FileNotFoundError(f"annotation directory not found: {annotation_dir}")
    return sorted(p for p in annotation_dir.rglob("*.xml") if p.is_file())


def _text(element, tag):
    child = element.find(tag)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def _read_bbox(object_element):
    bndbox = object_element.find("bndbox")
    if bndbox is None:
        return None
    values = []
    for tag in ("xmin", "ymin", "xmax", "ymax"):
        value = _text(bndbox, tag)
        if value is None:
            return None
        values.append(int(value))
    return values


def build_database(annotation_dir=DEFAULT_ANNOTATION_DIR, template_path=None):
    database = _initial_database(template_path)
    stats = {
        "xml_files": 0,
        "objects_seen": 0,
        "objects_written": 0,
        "missing_name": 0,
        "missing_bbox": 0,
        "missing_mask_path": 0,
        "parse_errors": 0,
        "unknown_categories": 0,
    }

    for xml_path in _find_xml_files(annotation_dir):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            stats["parse_errors"] += 1
            continue

        stats["xml_files"] += 1
        file_name = _text(root, "filename") or xml_path.stem

        for object_element in root.findall("object"):
            stats["objects_seen"] += 1
            pest_id = _text(object_element, "name")
            if pest_id is None:
                stats["missing_name"] += 1
                continue

            bbox = _read_bbox(object_element)
            if bbox is None:
                stats["missing_bbox"] += 1
                continue

            mask_path = _text(object_element, "mask_path")
            if mask_path is None:
                stats["missing_mask_path"] += 1
                continue

            if pest_id not in database:
                database[pest_id] = {"英文名": "", "中文名": "", "in_file": []}
                stats["unknown_categories"] += 1

            database[pest_id]["in_file"].append(
                {
                    "file_name": file_name,
                    "bbox": bbox,
                    "mask_path": mask_path,
                }
            )
            stats["objects_written"] += 1

    return database, stats


def write_database(database, output_path):
    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump([database], f, ensure_ascii=False)


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build pest24_database.json from Pest24 XML annotations with mask_path entries."
    )
    parser.add_argument("--annotation-dir", default=str(DEFAULT_ANNOTATION_DIR))
    parser.add_argument(
        "--template",
        default=None,
        help="Optional template JSON. If omitted, the built-in pest24 class table is used.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--verify-against", default=None)
    args = parser.parse_args(argv)

    database, stats = build_database(args.annotation_dir, args.template)
    write_database(database, args.output)


    if args.verify_against:
        generated = _load_json(args.output)
        expected = _load_json(args.verify_against)
        if generated != expected:
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
