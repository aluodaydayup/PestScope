import csv
import json
import os

import os
import re  
from openai import OpenAI


def get_api_output(sys_prompt, content):
    client = OpenAI(
        api_key='',
        base_url="https://pro.xiaoai.plus/v1",
    )

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"{sys_prompt}"},
            {"role": "user", "content": f"{content}"}
        ]
    )
    return completion.choices[0].message.content


def test(sys_prompt, content):
    completion = [sys_prompt, content]
    return completion


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


root_path = r"pest\DataProcessing\PestSourceDataToFeatureJson"
out_path = r"pest\DataProcessing\pest_topic_sentence.json"
output_data = []
for root, dirs, files in os.walk(root_path):
    for file in files:
        file_name = file.split('.')[0].split('_')[0]
        file_path = os.path.join(root, file)
        with open(file_path, 'r', encoding='utf-8') as json_file:
            dictionary = json.load(json_file)

            sys_prompt_fh = (
                f'你是一名专业的害虫形态学专家。回答始终用中文。'
                f'将的{file_name}分布与危害尽可能多的细致划分为几个要点，要求如下：'
                f'1. 尽可能详细划分为多个要点，每个要点清晰可区分。'
                f'2. 每个要点只用一个简短句子描述。'
                f'3.划分后的要点需覆盖段落中的所有内容，不遗漏信息。'
                f'4. 每个句子中都需包含“{file_name}”这个害虫名称，且“{file_name}”这个害虫名称要放在句首还要保持句子通顺。'
                f'5、将要点以python列表形式输出，每个要点是列表中的一个元素，只输出列表内容，其他任何内容都不要输出。'
            )

            sys_prompt_xt = (
                f'你是一名专业的害虫形态学专家。回答始终用中文。'
                f'将的{file_name}形态特征尽可能多的细致划分为几个要点，要求如下：'
                f'1. 尽可能详细划分为多个要点，每个要点清晰可区分。'
                f'2. 每个要点只用一个简短句子描述。'
                f'3.划分后的要点需覆盖段落中的所有内容，不遗漏信息。'
                f'4. 每个句子中都需包含“{file_name}”这个害虫名称，且“{file_name}”这个害虫名称要放在句首还要保持句子通顺。'
                f'5、将要点以python列表形式输出，每个要点是列表中的一个元素，只输出列表内容，其他任何内容都不要输出。'
            )

            sys_prompt_fs = (
                f'你是一名专业的害虫形态学专家。回答始终用中文。'
                f'将的{file_name}发生规律尽可能多的细致划分为几个要点，要求如下：'
                f'1. 尽可能详细划分为多个要点，每个要点清晰可区分。'
                f'2. 每个要点只用一个简短句子描述。'
                f'3.划分后的要点需覆盖段落中的所有内容，不遗漏信息。'
                f'4. 每个句子中都需包含“{file_name}”这个害虫名称，且“{file_name}”这个害虫名称要放在句首还要保持句子通顺。'
                f'5、将要点以python列表形式输出，每个要点是列表中的一个元素，只输出列表内容，其他任何内容都不要输出。'
            )

            sys_prompt_fz = (
                f'你是一名专业的害虫形态学专家。回答始终用中文。'
                f'将的{file_name}防治相关内容尽可能多的细致划分为几个要点，要求如下：'
                f'1. 尽可能详细划分为多个要点，每个要点清晰可区分。'
                f'2. 每个要点只用一个简短句子描述。'
                f'3. 划分后的要点需覆盖段落中的所有内容，不遗漏信息。'
                f'4. 每个句子中都需包含“{file_name}”这个害虫名称，且“{file_name}”这个害虫名称要放在句首还要保持句子通顺。'
                f'5、每个要点必须是具体的防治措施，有数值数据就要保留数值数据。'
                f'6、将要点以python列表形式输出，每个要点是列表中的一个元素，只输出列表内容，其他任何内容都不要输出。'
            )

            sys_prompt_cc = (
                f'你是一名专业的害虫形态学专家。回答始终用中文。'
                f'将的{file_name}成虫的形态特征尽可能多的细致划分为几个要点，要求如下：'
                f'1. 尽可能详细划分为多个要点，每个要点清晰可区分。'
                f'2. 每个要点只用一个简短句子描述。'
                f'3.划分后的要点需覆盖段落中的所有内容，不遗漏信息。'
                f'4. 每个句子中都需包含“{file_name}”这个害虫名称，且“{file_name}”这个害虫名称要放在句首还要保持句子通顺。'
                f'5、将要点以python列表形式输出，每个要点是列表中的一个元素，只输出列表内容，其他任何内容都不要输出。'
            )

            content_fh = dictionary['分布与危害']
            content_xt = dictionary['形态特征']
            content_fs = dictionary['发生规律']
            content_fz = dictionary['防治方法']
            content_cc = dictionary['成虫的形态特征']

            result_fh = get_api_output(sys_prompt_fh, content_fh)
            result_xt = get_api_output(sys_prompt_xt, content_xt)
            result_fs = get_api_output(sys_prompt_fs, content_fs)
            result_fz = get_api_output(sys_prompt_fz, content_fz)
            result_cc = get_api_output(sys_prompt_cc, content_cc)

            result_fh_str = clean_model_output(result_fh)
            result_xt_str = clean_model_output(result_xt)
            result_fs_str = clean_model_output(result_fs)
            result_fz_str = clean_model_output(result_fz)
            result_cc_str = clean_model_output(result_cc)

            json_data = {
                file_name: {
                    '分布与危害': result_fh_str,
                    '形态特征': result_xt_str,
                    '发生规律': result_fs_str,
                    '防治方法': result_fz_str,
                    '成虫的形态特征': result_cc_str
                }
            }

            output_data.append(json_data)
            json_file_path = out_path 

            with open(json_file_path, 'w', encoding='utf-8') as js_file:
                json.dump(output_data, js_file, ensure_ascii=False, indent=4)
