import os
import cv2
import random
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.SAM.utils.transforms import ResizeLongestSide
from dataset.utils.grefer import G_REFER
from dataset.utils.refcoco_refer import REFER
from tools.utils import DEFAULT_IMAGE_TOKEN
from dataset.utils.utils import ANSWER_LIST, SEG_QUESTIONS
from PIL import Image
import json
from itertools import combinations

SEED = 42
random.seed(SEED)            # Python random
SHORT_QUESTION_LIST_PEST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "你能在这张图片中分割出{class_name}吗？请输出分割掩码并给出害虫名。",
    DEFAULT_IMAGE_TOKEN + "\n" + "请在这张图片中分割出{class_name}。请输出分割掩码并给出害虫名。",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "这张图片中的{class_name}在哪里？请返回分割掩码。"
]
LONG_QUESTION_LIST_PEST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} 请给出分割掩码。",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} 请输出分割掩码。",
]
ANSWER_LIST_PEST = [
    "是 [SEG]。",
    "分割结果是 [SEG]。",
    "[SEG]。",
]
EXPLANATORY_QUESTION_LIST_PEST = [
    "请输出分割掩码并给出害虫名。",
    "请输出分割掩码并给出害虫名。",
    "请输出分割掩码并给出具体的害虫名。",
]
ANSWER_LIST_MODE4_TEMPLATE_PEST = [
    "{class_name} [SEG]",
    "{class_name}:[SEG]",
    "{class_name}的掩码是[SEG]",
    "{class_name}的分割掩码是[SEG]",
    "{class_name}的指代是[SEG]"
]
ANSWER_LIST_MODE4_TEMPLATE_NON_PEST = [
    "{class_name}:[NON]",
]
ANSWER_LIST_MODE4_START_PEST = [
    "当然,",
    "当然,",
    "当然,"
]
ANSWER_LIST_MODE4_END = [
    "。", "。", "。", "。", "。"
]
def get_pest_mask(segment_list):
    # 
    # 
    merged_mask = None
    # 
    for mask_file in segment_list:
        if "\\" in mask_file:
            mask_basename = mask_file.split('\\')[-1]  # 
        elif "/" in mask_file:
            mask_basename = mask_file.split('/')[-1]  # 
        else:
            mask_basename = mask_file
        mask_root_path = "/home /pycharm_workspace/SAM2/pest24_data/Pest24/mask_image/"
        mask_file = os.path.join(mask_root_path, mask_basename)
        # 
        mask_image = Image.open(mask_file)
        mask_array = np.array(mask_image)
        mask_array = (mask_array > 0).astype(np.uint8)
        if merged_mask is None:
            merged_mask = mask_array
        else:
            merged_mask = np.bitwise_or(merged_mask, mask_array)
    mask = merged_mask.astype(np.int32)
    return mask
def get_mask_from_json_pest_mul(json_path, img):
    try:
        with open(json_path, "r") as r:
            anno = json.loads(r.read())
    except:
        with open(json_path, "r", encoding="cp1252") as r:
            anno = json.loads(r.read())

    pest_name_list = anno["ann"]

    inform = anno["shapes"]
    comments = anno["text"]
    is_sentence = anno["is_sentence"]
    annotations = anno['ann']
    segment_list = []
    pest_list = []
    bbox_list = []

    non_segment_list = []
    non_pest_list = []
    non_bbox_list = []
    for annotation in annotations:
        if "is_target" in annotation:
            segment = annotation["segmentation"]
            pest_name = annotation["pest_name"]
            bbox = annotation["bbox"]
            if annotation["is_target"]:
                mask = get_pest_mask(segment)
                segment_list.append(mask)
                pest_list.append(pest_name)
                bbox_list.append(bbox)
            else:
                mask = get_pest_mask(segment)
                non_segment_list.append(mask)
                non_pest_list.append(pest_name)
                non_bbox_list.append(bbox)
        else:
            segment = annotation["segmentation"]
            pest_name = annotation["pest_name"]
            bbox = annotation["bbox"]
            mask = get_pest_mask(segment)
            segment_list.append(mask)
            pest_list.append(pest_name)
            bbox_list.append(bbox)

    return segment_list, comments, is_sentence, pest_list, bbox_list,non_segment_list, non_pest_list, non_bbox_list


class ReferSegmDataset(torch.utils.data.Dataset):
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=500 * 8 * 2 * 10,
                 precision: str = "fp32", image_size: int = 224, num_classes_per_sample: int = 3,
                 refer_segm_data="refcoco||refcoco+||refcocog||refclef", validation=False, split='train',explanatory=0.1,
                 image_root_path=None,
                 reason_seg_data_add=None,
                 reason_image_root_path=None,
                 random_sampling=False, inference=False):
        self.epoch_samples = epoch_samples
        self.num_classes_per_sample = num_classes_per_sample

        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)

        self.question_templates = SEG_QUESTIONS
        self.answer_list = ANSWER_LIST
        self.begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        self.validation = validation
        self.split = split
        self.random_sampling = random_sampling
        # self.initialize_refer_segm_data(refer_segm_data, inference)

        self.image_root_path = image_root_path
        self.reason_seg_data_add = reason_seg_data_add
        self.reason_image_root_path = reason_image_root_path
        self.short_question_list = SHORT_QUESTION_LIST_PEST
        self.long_question_list = LONG_QUESTION_LIST_PEST
        self.answer_list = ANSWER_LIST_PEST
        self.class_name_answer = ANSWER_LIST_MODE4_TEMPLATE_PEST
        self.class_name_answer_non = ANSWER_LIST_MODE4_TEMPLATE_NON_PEST
        self.explanatory = explanatory
        images = []
        jsons = []

        priority = {"simple": 0, "medium": 1, "dif": 2}

        def sort_key(filename):
            parts = filename.split("_", 1)
            prefix = parts[0]
            return (priority.get(prefix, 3), filename)

        whole_path = os.path.join(dataset_dir, 'reason_seg', "ReasonSeg", "train")

        if not os.path.exists(whole_path):
            print(f"路径不存在: {whole_path}")
        image_root_path = self.image_root_path
        for root, dirs, files in os.walk(whole_path):
            sorted_files = sorted(files, key=sort_key)

            for file in sorted_files:
                json_path = os.path.join(root, file)
                assert os.path.exists(json_path)
                jsons.append(json_path)
                file_name = file.split(".")[0]
                image_file_name = str(file_name) + ".jpg"

                image_path = os.path.join(image_root_path, image_file_name)
                if not os.path.exists(image_path):
                    image_root_path2 = "/home /pycharm_workspace/SAM2/pest24_data/Pest24/mask_image/"
                    image_path = os.path.join(image_root_path2, image_file_name)
                assert os.path.exists(image_path)
                images.append(image_path)
        if self.reason_seg_data_add is not None:
            whole_path_rs_add = os.path.join(self.reason_seg_data_add, 'reason_seg', "ReasonSeg", "train")
            if not os.path.exists(whole_path_rs_add):
                print(f"路径不存在: {whole_path_rs_add}")

            reason_image_root_path = self.reason_image_root_path
            add_json = []
            add_images = []
            for root, dirs, files in os.walk(whole_path_rs_add):
                sorted_files = sorted(files, key=sort_key)

                for file in sorted_files:
                    json_path = os.path.join(root, file)
                    assert os.path.exists(json_path)
                    add_json.append(json_path)
                    file_name = file.split(".")[0]
                    image_file_name = str(file_name) + ".jpg"

                    image_path = os.path.join(reason_image_root_path, image_file_name)
                    assert os.path.exists(image_path)
                    add_images.append(image_path)
            jsons = jsons + add_json
            images = images + add_images

        self.refer_segm_data = (images, jsons)

        print("number of reason_seg samples: ", len(images))
        print("number of reason_seg samples jsons: ", len(jsons))
        if explanatory != -1:
            self.explanatory_question_list = EXPLANATORY_QUESTION_LIST_PEST


    def __len__(self):
        return len(self.refer_segm_data[0])
    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def _set_len(self, length):
        self.epoch_samples = length


    def __getitem__(self, idx):

        refer_seg_ds = self.refer_segm_data
        images, jsons = refer_seg_ds
        idx = idx if (self.validation or not self.random_sampling) else random.randint(0, len(images) - 1)

        image_path = images[idx]
        json_path = jsons[idx]
        image = cv2.imread(image_path)

        image_cl = image
        image_cl = cv2.cvtColor(image_cl, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        ori_size = image.shape[:2]
        # preprocess image for clip
        image_clip = self.global_enc_processor.preprocess(image_cl, return_tensors="pt")[
            "pixel_values"
        ][0]

        mask, sents, is_sentence, pest_list, bbox_list, non_segment_list, non_pest_list, non_bbox_list = get_mask_from_json_pest_mul(
            json_path, image)

        all_mask_list = mask

        # 
        assert len(mask) == len(pest_list)

        if not is_sentence:
            # 
            one_choice = random.choice([1, 2, 3])
            if len(pest_list) <= one_choice:
                one_choice = 1
            print('one_choice:', one_choice)


            if one_choice == 3:
                all_ms = mask
                al_pest_list = pest_list
                al_bbox_list = bbox_list
                al_segment_list = mask
                index_list = [index for index, value in enumerate(all_ms)]

                is_choice = random.sample(index_list, 3)

                mask = [mask[random_count] for random_count in is_choice]
                pest_list = [pest_list[random_count] for random_count in is_choice]

                non_mask_idx = [i for i in index_list if i not in is_choice]
                if len(non_mask_idx) != 0:
                    non_mask_list = []
                    for non_i in non_mask_idx:
                        non_mask_list.append(all_ms[non_i])
                        non_pest_list.append(al_pest_list[non_i])
                        non_bbox_list.append(al_bbox_list[non_i])
                        non_segment_list.append(al_segment_list[non_i])
                    non_merged_mask_ = non_mask_list
                else:
                    non_merged_mask_ = []
            elif one_choice == 2:
                all_ms = mask
                al_pest_list = pest_list
                al_bbox_list = bbox_list
                al_segment_list = mask
                index_list = [index for index, value in enumerate(all_ms)]

                is_choice = random.sample(index_list, 2)

                mask = [mask[random_count] for random_count in is_choice]
                pest_list = [pest_list[random_count] for random_count in is_choice]

                non_mask_idx = [i for i in index_list if i not in is_choice]
                if len(non_mask_idx) != 0:
                    non_mask_list = []
                    for non_i in non_mask_idx:
                        non_mask_list.append(all_ms[non_i])
                        non_pest_list.append(al_pest_list[non_i])
                        non_bbox_list.append(al_bbox_list[non_i])
                        non_segment_list.append(al_segment_list[non_i])
                    
                    non_merged_mask_ = non_mask_list
                else:
                    non_merged_mask_ = []


            else:
                assert len(pest_list) == len(mask)
                is_choice = random.randint(0, len(pest_list) - 1)
                all_ms = mask
                al_pest_list = pest_list
                al_bbox_list = bbox_list
                al_segment_list = mask
                mask = [all_ms[is_choice]]
                non_mask_idx = [i for i in range(len(pest_list)) if i != is_choice]
                if len(non_mask_idx) != 0:
                    non_mask_list = []
                    for non_i in non_mask_idx:
                        non_mask_list.append(all_ms[non_i])
                        non_pest_list.append(al_pest_list[non_i])
                        non_bbox_list.append(al_bbox_list[non_i])
                        non_segment_list.append(al_segment_list[non_i])

                    non_merged_mask_ = non_mask_list
                else:
                    non_merged_mask_ = []
                pest_list = [pest_list[is_choice]]

        else:


            assert len(pest_list) == len(mask)
            assert len(non_segment_list) == len(non_bbox_list) == len(non_pest_list)
            if len(non_segment_list) != 0:
                non_merged_mask_ = non_segment_list
            else:
                non_merged_mask_ = []

        sampled_sents = [sents]
        sampled_masks = [np.array(m).astype(np.float32) for m in mask]

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        image_name = image_path.split("/")[-1]
        # 
        choice = 0
        questions = []
        answers = []
        # 
        pest_str = ', '.join(pest_list)

        for text in sampled_sents:
            if is_sentence:
                question_template = random.choice(self.long_question_list)
                questions.append(question_template.format(sent=text[0]))
            else:
                question_template = random.choice(self.short_question_list)
                questions.append(question_template.format(class_name=pest_str.lower()))
            # add explanation if applicable
            img_name = image_path.split("/")[-1]

            if self.explanatory != -1:
                if choice == 0:  # [SEG] token
                    ans_start = random.choice(ANSWER_LIST_MODE4_START_PEST)
                    ans_end = random.choice(ANSWER_LIST_MODE4_END)
                    seg_token_parts = []
                    non_token_parts = []
                    if is_sentence:
                        for pest_name in pest_list:
                            question_template = random.choice(self.class_name_answer)
                            question_template = question_template.format(class_name=pest_name)
                            seg_token_parts.append(question_template)

                        for non_pest_name in non_pest_list:
                            ans_template = random.choice(self.class_name_answer_non)
                            ans_template = ans_template.format(class_name=non_pest_name)
                            non_token_parts.append(ans_template)

                        answers.append(
                            ans_start + " " + ", ".join(seg_token_parts) + ans_end + " " + ", ".join(non_token_parts)
                        )
                    else:
                        for pest_name in pest_list:
                            question_template = random.choice(self.class_name_answer)
                            question_template = question_template.format(class_name=pest_name)
                            seg_token_parts.append(question_template)

                        for non_pest_name in non_pest_list:
                            ans_template = random.choice(self.class_name_answer_non)
                            ans_template = ans_template.format(class_name=non_pest_name)
                            non_token_parts.append(ans_template)

                        answers.append(
                            ans_start + " " + ", ".join(seg_token_parts) + ans_end + " " + ", ".join(non_token_parts)
                        )

                else:
                    raise ValueError("Not implemented yet.")
            else:
                answers.append(random.choice(self.answer_list))

            conversations = []
            conv = conversation_lib.default_conversation.copy()
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

            i = 0
            while i < len(questions):
                conv.messages = []
                conv.append_message(conv.roles[0], questions[i])

                conv.append_message(conv.roles[1], answers[i])

                # conv.append_message(conv.roles[1],random.choice(ANSWER_LIST_PEST))

                conversations.append(conv.get_prompt())
                i += 1
            # print(conversations)
        image = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        image_name = image_path.split("/")[-1]

        masks = np.stack(sampled_masks, axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.IGNORE_LABEL
        if len(non_merged_mask_) != 0:
            non_merged_mask = [np.array(m).astype(np.float32) for m in non_merged_mask_]
            non_masks = np.stack(non_merged_mask, axis=0)
            non_masks = torch.from_numpy(non_masks)
        else:
            non_masks = torch.zeros_like(masks)
            non_masks = non_masks[:0]
            # print(non_masks.size(0))
        print(conversations)
        global_enc_img = image_clip
        grounding_enc_img = image
        bboxes = None
        image_resize = resize
        selected_labels = sampled_sents
        return (image_path, global_enc_img, grounding_enc_img, bboxes, conversations, masks, label,
                image_resize, questions, selected_labels,non_masks)


class ReferSegmDatasetVal(torch.utils.data.Dataset):
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=500 * 8 * 2 * 10,
                 precision: str = "fp32", image_size: int = 224, num_classes_per_sample: int = 3,
                 refer_segm_data="refcoco||refcoco+||refcocog||refclef", validation=False, split='train',explanatory=0.1,
                 image_root_path=None,
                 reason_seg_data_add=None,
                 reason_image_root_path=None,
                 random_sampling=False, inference=False):
        self.epoch_samples = epoch_samples
        self.num_classes_per_sample = num_classes_per_sample

        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)

        self.question_templates = SEG_QUESTIONS
        self.answer_list = ANSWER_LIST
        self.begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        self.validation = validation
        self.split = split
        self.random_sampling = random_sampling
        # self.initialize_refer_segm_data(refer_segm_data, inference)

        self.image_root_path = image_root_path
        self.reason_seg_data_add = reason_seg_data_add
        self.reason_image_root_path = reason_image_root_path
        self.short_question_list = SHORT_QUESTION_LIST_PEST
        self.long_question_list = LONG_QUESTION_LIST_PEST
        self.answer_list = ANSWER_LIST_PEST
        self.class_name_answer = ANSWER_LIST_MODE4_TEMPLATE_PEST
        self.class_name_answer_non = ANSWER_LIST_MODE4_TEMPLATE_NON_PEST
        self.explanatory = explanatory

        images = []
        jsons = []
        if self.reason_seg_data_add is not None:
            whole_path = os.path.join(reason_seg_data_add, 'reason_seg', "ReasonSeg", "val")
            if not os.path.exists(whole_path):
                print(f"路径不存在: {whole_path}")
            image_root_path = self.reason_image_root_path

            for root, dirs, files in os.walk(whole_path):
                for file in files:
                    json_path = os.path.join(root, file)
                    assert os.path.exists(json_path)

                    jsons.append(json_path)
                    file_name = file.split(".")[0]
                    image_file_name = str(file_name) + ".jpg"

                    image_path = os.path.join(image_root_path, image_file_name)

                    assert os.path.exists(image_path), f"图像文件不存在: {image_path}"
                    images.append(image_path)
        else:
            whole_path = os.path.join(dataset_dir, 'reason_seg', "ReasonSeg", "val")
            if not os.path.exists(whole_path):
                print(f"路径不存在: {whole_path}")
            image_root_path = self.image_root_path

            for root, dirs, files in os.walk(whole_path):
                for file in files:
                    json_path = os.path.join(root, file)
                    assert os.path.exists(json_path)

                    jsons.append(json_path)
                    file_name = file.split(".")[0]
                    image_file_name = str(file_name) + ".jpg"

                    image_path = os.path.join(image_root_path, image_file_name)

                    assert os.path.exists(image_path)
                    images.append(image_path)
        assert len(images) == len(jsons)
        self.jsons = jsons
        self.images = images

        self.data_type = "reason_seg"


    def __len__(self):
        return len(self.images)

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def _set_len(self, length):
        self.epoch_samples = length

    def __getitem__(self, idx):

        idx = idx if (self.validation or not self.random_sampling) else random.randint(0, len(self.images) - 1)

        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            is_sentence = False
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)


            image_cl = image
            image_cl = cv2.cvtColor(image_cl, cv2.COLOR_BGR2RGB)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            json_path = self.jsons[idx]

            mask_json, sampled_sents, is_sentence, pest_list, bbox_list, non_segment_list, non_pest_list, non_bbox_list = get_mask_from_json_pest_mul(
                json_path, image)

            if not is_sentence:
                all_bbox_list = bbox_list
                all_pest_list = pest_list
                all_mask_list = mask_json

                one_choice = random.choice([1, 2, 3])
                if one_choice >= len(all_pest_list):
                    one_choice = 1
                print('one_choice:', one_choice)
                if one_choice == 2:
                    index_list = [index for index, value in enumerate(pest_list)]
                    pest_combinations = list(combinations(index_list, 2))
                    w_list = []
                    for pc in pest_combinations:
                        w_list.append(1)
                    assert len(w_list) == len(pest_combinations)
                    is_choice = random.choices(pest_combinations, weights=w_list, k=1)[0]

                    non_mask_idx = [i for i in range(len(all_mask_list)) if i not in is_choice]
                    non_mask_list = []
                    for non_i in non_mask_idx:
                        non_mask_list.append(all_mask_list[non_i])
                        non_pest_list.append(all_pest_list[non_i])
                        non_bbox_list.append(all_bbox_list)

                    non_merged_mask_ = non_mask_list


                    mask_json = [mask_json[ic] for ic in is_choice]
                    pest_list = [pest_list[ic] for ic in is_choice]
                    bbox_list = [bbox_list[ic] for ic in is_choice]
                elif one_choice == 3:
                    index_list = [index for index, value in enumerate(pest_list)]
                    pest_combinations = list(combinations(index_list, 3))
                    w_list = []
                    for pc in pest_combinations:
                        w_list.append(1)
                    assert len(w_list) == len(pest_combinations)
                    is_choice = random.choices(pest_combinations, weights=w_list, k=1)[0]

                    non_mask_idx = [i for i in range(len(all_mask_list)) if i not in is_choice]

                    non_mask_list = []
                    for non_i in non_mask_idx:
                        non_mask_list.append(all_mask_list[non_i])
                        non_pest_list.append(all_pest_list[non_i])
                        non_bbox_list.append(all_bbox_list)
                    non_merged_mask_ = non_mask_list

                    mask_json = [mask_json[ic] for ic in is_choice]
                    pest_list = [pest_list[ic] for ic in is_choice]
                    bbox_list = [bbox_list[ic] for ic in is_choice]

                elif one_choice == 1:
                    is_choice = random.choice(range(len(pest_list)))
                    non_mask_idx = [i for i in range(len(pest_list)) if i != is_choice]

                    non_mask_list = []
                    for non_i in non_mask_idx:
                        non_mask_list.append(all_mask_list[non_i])
                        non_pest_list.append(all_pest_list[non_i])
                        non_bbox_list.append(all_bbox_list)
                    non_merged_mask_ = non_mask_list
                    mask_json = [mask_json[is_choice]]
                    pest_list = [pest_list[is_choice]]
                    bbox_list = [bbox_list[is_choice]]

                sampled_sents = [sampled_sents]
            else:
                all_bbox_list = bbox_list + non_bbox_list
                all_pest_list = pest_list + non_pest_list
                all_mask_list = mask_json + non_segment_list

                assert len(non_segment_list) == len(non_bbox_list) == len(non_pest_list)
                if len(non_segment_list) != 0:
                    non_merged_mask_ = non_segment_list
                else:
                    non_merged_mask_ = []
                sampled_sents = sampled_sents

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n {} 请给出分割掩码。".format(text),
                )

                answers = []
                ans_start = random.choice(ANSWER_LIST_MODE4_START_PEST)
                ans_end = random.choice(ANSWER_LIST_MODE4_END)
                seg_token_parts = []
                non_token_parts = []
                for pest_name in pest_list:
                    question_template = random.choice(self.class_name_answer)
                    question_template = question_template.format(class_name=pest_name)
                    seg_token_parts.append(question_template)

                for non_pest_name in non_pest_list:
                    ans_template = random.choice(self.class_name_answer_non)
                    ans_template = ans_template.format(class_name=non_pest_name)
                    non_token_parts.append(ans_template)

                answers.append(
                    ans_start + " " + ", ".join(seg_token_parts) + ans_end + " " + ", ".join(non_token_parts)
                )

                conv.append_message(conv.roles[1], answers[0])
                # conv.append_message(conv.roles[1], "[SEG].")
            else:
                text = ",".join(pest_list)
                conv.append_message(
                    conv.roles[0],
                    DEFAULT_IMAGE_TOKEN
                    + "\n  {}在图片中的什么地方? 请给出分割掩码。".format(
                        text
                    ),
                )
                answers = []
                ans_start = random.choice(ANSWER_LIST_MODE4_START_PEST)
                ans_end = random.choice(ANSWER_LIST_MODE4_END)
                seg_token_parts = []
                non_token_parts = []
                for pest_name in pest_list:
                    question_template = random.choice(self.class_name_answer)
                    question_template = question_template.format(class_name=pest_name)
                    seg_token_parts.append(question_template)

                for non_pest_name in non_pest_list:
                    ans_template = random.choice(self.class_name_answer_non)
                    ans_template = ans_template.format(class_name=non_pest_name)
                    non_token_parts.append(ans_template)

                answers.append(
                    ans_start + " " + ", ".join(seg_token_parts) + ans_end + " " + ", ".join(non_token_parts)
                )

                conv.append_message(conv.roles[1], answers[0])
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.global_enc_processor.preprocess(image_cl, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        masks = mask_json

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.IGNORE_LABEL

        if len(non_merged_mask_) != 0:
            non_merged_mask = non_merged_mask_
            non_merged_mask = np.stack(non_merged_mask, axis=0)
            non_merged_mask = torch.from_numpy(non_merged_mask)
        else:
            non_merged_mask = torch.zeros_like(masks)
            non_merged_mask = non_merged_mask[:0]
        inference = True
        print(conversations)
        global_enc_img = image_clip
        grounding_enc_img = image
        bboxes = None
        image_resize = resize
        selected_labels = sampled_sents
        return (image_path, global_enc_img, grounding_enc_img, bboxes, conversations,
                masks, labels,image_resize,None,None,inference,pest_list,bbox_list,
                all_bbox_list,all_pest_list,all_mask_list,non_merged_mask)



class ReferSegmDatasetassdsd(torch.utils.data.Dataset):
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 1024
    IGNORE_LABEL = 255

    def __init__(self, dataset_dir, tokenizer, global_image_encoder, epoch_samples=500 * 8 * 2 * 10,
                 precision: str = "fp32", image_size: int = 224, num_classes_per_sample: int = 3,
                 refer_segm_data="refcoco||refcoco+||refcocog||refclef", validation=False, split='train',
                 random_sampling=False, inference=False):
        self.epoch_samples = epoch_samples
        self.num_classes_per_sample = num_classes_per_sample

        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)

        self.question_templates = SEG_QUESTIONS
        self.answer_list = ANSWER_LIST
        self.begin_str = f"""The {DEFAULT_IMAGE_TOKEN} provides an overview of the picture.\n"""
        self.validation = validation
        self.split = split
        self.initialize_refer_segm_data(refer_segm_data, inference)
        self.random_sampling = random_sampling

    def initialize_refer_segm_data(self, refer_segm_data, inference=False):

        dataset_dir = os.path.join(self.dataset_dir, "Refer_Segm")
        self.refer_seg_ds_list = refer_segm_data.split("||")
        # ['refclef', 'refcoco', 'refcoco+', 'refcocog']
        self.refer_segm_data = {}

        for dataset_name in self.refer_seg_ds_list:
            splitBy = "umd" if dataset_name == "refcocog" else "unc"
            refer_api = G_REFER(dataset_dir, dataset_name, splitBy) if dataset_name == "grefcoco" else\
                REFER(dataset_dir, dataset_name, splitBy)
            ref_ids_train = refer_api.getRefIds(split=self.split)
            images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
            refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)
            refer_seg_ds = {
                "images": self.load_images(refer_api, images_ids_train, dataset_dir, dataset_name, inference=inference),
                "annotations": refer_api.Anns,
                "img2refs": self.create_img_to_refs_mapping(refs_train)
            }

            print(f"dataset {dataset_name} (refs {splitBy}) ({self.split} split) has {len(refer_seg_ds['images'])} "
                  f"images and {len(refer_seg_ds['annotations'])} annotations.")
            print(f'\033[92m----SEG-{"Val" if self.validation else "Train"}:'
                  f' Loaded ReferSeg - {dataset_name} dataset ----\033[0m')

            self.refer_segm_data[dataset_name] = refer_seg_ds

    def load_images(self, refer_api, images_ids_train, dataset_dir, dataset_name, inference=False):
        images = []
        loaded_images = refer_api.loadImgs(image_ids=images_ids_train)
        # Limiting images to 1000(optional) for validation
        loaded_images = loaded_images[:1000] if (self.validation and not inference) else loaded_images
        for item in loaded_images:
            item = item.copy()
            if dataset_name == 'refclef':
                item["file_name"] = os.path.join(dataset_dir, "images", "saiapr_tc-12", item["file_name"])
            else:
                item["file_name"] = os.path.join(dataset_dir.replace("Refer_Segm/", ""), "coco_2014/train2014",
                                                 item["file_name"])
            images.append(item)
        return images

    def create_img_to_refs_mapping(self, refs_train):
        img2refs = {}
        for ref in refs_train:
            img2refs[ref["image_id"]] = img2refs.get(ref["image_id"], []) + [ref, ]
        return img2refs

    def __len__(self):
        return self.epoch_samples

    def _set_len(self, length):
        self.epoch_samples = length

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN) / self.IMG_STD
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h))
        return x

    def create_conversations(self, labels):
        questions = []
        answers = []
        for i, label in enumerate(labels):
            label = label.strip()
            assert len(label.split("||")) == 1
            question_template = random.choice(self.question_templates)
            questions.append(question_template.format(class_name=label.lower()))
            answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                question = self.begin_str + question
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], answer)
        conversations.append(conv.get_prompt())
        return questions, conversations

    def __getitem__(self, idx):
        dataset_idx = random.randint(0, len(self.refer_seg_ds_list) - 1)
        dataset_name = self.refer_seg_ds_list[dataset_idx]
        refer_seg_ds = self.refer_segm_data[dataset_name]
        images = refer_seg_ds["images"]
        annotations = refer_seg_ds["annotations"]
        img2refs = refer_seg_ds["img2refs"]
        idx = idx if (self.validation or not self.random_sampling) else random.randint(0, len(images) - 1)
        image_info = images[idx]
        image_id = image_info["id"]
        refs = img2refs[image_id]
        if len(refs) == 0:
            return self.__getitem__(0)

        sents = []
        ann_ids = []
        for ref in refs:
            for sent in ref["sentences"]:
                text = sent["sent"]
                sents.append(text)
                ann_ids.append(ref["ann_id"])
        if len(sents) >= self.num_classes_per_sample:
            sampled_inds = np.random.choice(
                list(range(len(sents))), size=self.num_classes_per_sample, replace=False
            )
        else:
            sampled_inds = list(range(len(sents)))
        sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
        # sampled_ann_ids = np.vectorize(ann_ids.__getitem__)(sampled_inds).tolist()
        sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
        selected_labels = sampled_sents

        # Load and process the image
        image_path = image_info["file_name"]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        global_enc_img = self.global_enc_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        image = self.transform.apply_image(image)
        image_resize = image.shape[:2]
        grounding_enc_img = self.grounding_enc_processor(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        # Generate questions and answers
        questions, conversations = self.create_conversations(selected_labels)

        flag = False
        masks = []
        for ann_id in sampled_ann_ids:
            if isinstance(ann_id, list):
                flag = True
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = annotations[ann_id_i]

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        else:
                            if type(ann["segmentation"][0]) == list:  # polygon
                                rle = mask.frPyObjects(
                                    ann["segmentation"], image_info["height"], image_info["width"], )
                            else:
                                rle = ann["segmentation"]
                                for i in range(len(rle)):
                                    if not isinstance(rle[i]["counts"], bytes):
                                        rle[i]["counts"] = rle[i]["counts"].encode()
                            m = mask.decode(rle)
                            m = np.sum(
                                m, axis=2
                            )  # sometimes there are multiple binary map (corresponding to multiple segs)
                            m = m.astype(np.uint8)  # convert to np.uint8
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = annotations[ann_id]

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = mask.decode(rle)
            m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)

        masks = np.stack(masks, axis=0)

        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.IGNORE_LABEL
        # set bboxes to None for segmentation datasets
        bboxes = None

        return (image_path, global_enc_img, grounding_enc_img, bboxes, conversations, masks, label,
                image_resize, questions, selected_labels)
