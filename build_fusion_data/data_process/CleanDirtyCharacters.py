import json
import re
import ast

FIELDS = ["分布与危害", "形态特征", "发生规律", "防治方法", "成虫的形态特征"]
def clean_model_output(text: str) -> str:

    if not text:
        return ""

    s = str(text)

    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s.strip())

    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\t", " ")
    s = s.replace("\n", " ")

    s = " ".join(s.split())

    return s

def parse_inner_list(value):

    if value is None:
        return []

    if isinstance(value, list):
        return [" ".join(str(x).split()) for x in value]

    s = str(value).strip()

    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()


    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, list):
            return [" ".join(str(x).split()) for x in obj]
    except Exception:
        pass


    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [" ".join(str(x).split()) for x in obj]
    except Exception:
        pass


    s_no_bracket = s.strip("[]")
    parts = re.split(r"[\n,]", s_no_bracket)
    items = []
    for p in parts:
        p = p.strip().strip("\"'")
        if p:
            items.append(" ".join(p.split()))

    if items:
        return items


    return [" ".join(s.split())]


def clean_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)


    for obj in data:

        for pest_name, pest_info in obj.items():
            if not isinstance(pest_info, dict):
                continue
            for field in FIELDS:
                if field in pest_info:
                    pest_info[field] = clean_model_output(pest_info[field])
                    pest_info[field] = parse_inner_list(pest_info[field])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":

    input_json = r"pest\DataProcessing\pest_topic_sentence.json"   
    output_json = r"pest\DataProcessing\pest_topic_sentence_clean.json" 

    clean_file(input_json, output_json)
