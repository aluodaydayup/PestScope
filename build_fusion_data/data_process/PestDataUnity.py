import json
import os


def load_class_mapping(class_path):

    with open(class_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cn_to_en = {}
    for item in data:
        cn = item.get("中文名")
        latin = item.get("英文名")
        if cn:
            cn_to_en[cn] = latin or ""
    return cn_to_en


def convert_pest_topic_to_pestsample(
    class_path: str,
    pest_topic_path: str,
    output_path: str,
    base_knowledge_path: str = "PestSourceDataToFeatureJson",

):
    cn_to_en = load_class_mapping(class_path)

    with open(pest_topic_path, "r", encoding="utf-8") as f:
        pest_topics = json.load(f)

    output = []
    idx = 1

    for entry in pest_topics:
        if not isinstance(entry, dict) or not entry:
            continue

        cn_name, bg_content = next(iter(entry.items()))

        en_name = cn_to_en.get(cn_name, "")

        knowledge_path = f"{base_knowledge_path}/{cn_name}"


        record = {
            str(idx): {
                "英文名": en_name,
                "中文名": cn_name,
                "知识路径": knowledge_path,
                "背景知识": bg_content,
            }
        }
        output.append(record)
        idx += 1

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)



if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    class_json_path = os.path.join(base_dir, "pest\DataProcessing\class.json")
    pest_topic_json_path = os.path.join(base_dir, "pest\DataProcessing\pest_topic_sentence_clean.json")
    output_json_path = os.path.join(base_dir, "pest\DataProcessing\pest_unity.json")

    convert_pest_topic_to_pestsample(
        class_path=class_json_path,
        pest_topic_path=pest_topic_json_path,
        output_path=output_json_path,
        base_knowledge_path="PestSourceDataToFeatureJson",
    )
