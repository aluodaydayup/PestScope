"""
train_ft.py - PestScope single-dataset training entry point

Fine-tunes PestScope on a selected pest reasoning segmentation dataset for agricultural visual question answering.
The training pipeline uses [SEG] for target pest masks and [NON] for exclusive non-target suppression.
"""
import csv
import os
import shutil
import sys
import time
import cv2
import tqdm
import random
import torch

import torch.distributed as dist

import argparse
import deepspeed
import numpy as np
import transformers
from functools import partial
from torch.utils.data import ConcatDataset
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.PestScope import GLaMMForCausalLM
from model.llava import conversation as conversation_lib

from dataset.dataset import custom_collate_fn,custom_collate_fn_val
from tools.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, AverageMeter, ProgressMeter, dict_to_cuda,
                         Summary, intersectionAndUnionGPU)

from dataset.gcg_datasets.GranDf_gcg_ds import GranDfDataset, OpenPsgGCGDataset, Flickr30kGCGDataset, RefCOCOgGCGDataset
from dataset.caption_datasets.COCO_Caption_ds import CocoCapDataset
from dataset.caption_datasets.LLavaInstruct_vqa_ds import LLaVAInstructDataset
from dataset.region_datasets.RefCOCO_VG_Region_ds import (RefCocoRegDataset, RefCocoGRegDataset, RefCocoPRegDataset,
                                                       VisualGenomeRegDataset)
from dataset.region_datasets.Flickr_Region_ds import Flickr30kRegDataset
from dataset.segm_datasets.Semantic_Segm_ds import SemanticSegmDataset
from dataset.segm_datasets.Pestseg_ds import ReferSegmDataset, ReferSegmDatasetVal
IMAGE_TOKEN_INDEX = -200

def parse_args(args):
    parser = argparse.ArgumentParser(description="PestScope Model Training")

    # Model-specific settings
    parser.add_argument("--version", default="MBZUAI/GLaMM-GranD-Pretrained")
    parser.add_argument("--vision_pretrained", default="./checkpoints/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--tune_mm_mlp_adapter", action="store_true")
    parser.add_argument("--freeze_mm_mlp_adapter", action="store_true")
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=True)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--image_size", default=1024, type=int, help="Image size for grounding image encoder")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--with_region", action="store_true", default=True)
    parser.add_argument("--mm_vision_select_layer", default=-2, type=int)
    parser.add_argument("--pretrain_mm_mlp_adapter", default="", type=str)
    parser.add_argument("--precision", default='bf16', type=str)

    # Dataset settings
    parser.add_argument("--use_cap_data", action="store_true", help="Use caption data")
    parser.add_argument("--use_reg_data", action="store_true", help="Use region data")
    parser.add_argument("--use_segm_data", action="store_true", help="Use segmentation data")
    # parser.add_argument("--no_sampling", action="store_true", default=False, help="Only one dataset finetuning, train on full length dataset.")
    parser.add_argument("--dataset_dir", default="./data", type=str)
    parser.add_argument(
        "--image_root_path", default="/home /pycharm_workspace/lisa/imgs", type=str
    )
    parser.add_argument("--reason_seg_data_add", default=None, type=str,help="推理分割数据的根路径")
    parser.add_argument("--reason_image_root_path", default=None, type=str,help="推理分割的图片路径")

    parser.add_argument("--seg_dataset", default="Semantic_Segm||Refer_Segm||RefCoco_GCG||PSG_GCG||Flickr_GCG||GranDf_GCG",
                        type=str, help="Choose from: Semantic_Segm, Refer_Segm, RefCoco_GCG, GranDf_GCG, PSG_GCG, Flickr_GCG")
    parser.add_argument("--segm_sample_rates", default="5,4,3,3,3,1", type=str)
    parser.add_argument("--reg_dataset", default="RefCoco_Reg||RefCocoG_Reg||RefCocoP_Reg||VisGen_Reg",
                        type=str, help="Choose from: RefCoco_Reg, RefCocoG_Reg, RefCocoP_Reg, VisGen_Reg, Flickr_Reg")
    parser.add_argument("--reg_sample_rates", default="1,1,1,1", type=str)
    parser.add_argument("--cap_dataset", default="CocoCap||LLaVaInstruct", type=str, help="Choose from: CocoCap, LLaVaInstruct")
    parser.add_argument("--cap_sample_rates", default="1,1", type=str)
    parser.add_argument("--semantic_segm_data", default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary", type=str)
    parser.add_argument("--refer_segm_data", default="refcoco||refcoco+||refcocog||refclef", type=str)
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)

    # Training settings
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--weight", default="", type=str)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=1, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")

    # Evaluation settings
    parser.add_argument("--val_dataset", default="RefCOCOgRegVal", type=str,
                        help="Choose from: CocoCapVal, RefCOCOgRegVal, VisGenomeRegVal, RefCOCOgSegmVal, PsgGCGVal, "
                             "RefCocoGCGVal, FlickrGCGVal")
    parser.add_argument("--mask_validation", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    # Experiment settings
    parser.add_argument("--log_base_dir", default="./output", type=str)
    parser.add_argument("--exp_name", default="PestScopeFinetuneOS", type=str)

    return parser.parse_args(args)


def initialize_environment(args):
    """ Set up logging and model directories. """
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        return SummaryWriter(args.log_dir)
    return None


def setup_tokenizer_and_special_tokens(args):
    """ Load tokenizer and add special tokens. """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    print('\033[92m' + "---- Initialized tokenizer from: {} ----".format(args.version) + '\033[0m')
    tokenizer.pad_token = tokenizer.unk_token

    if not args.pretrained:
        if args.use_mm_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        # modifications specific for regions
        reg_tokens = ['<bbox>', '<point>']
        # PestScope uses [SEG] for target pest masks and [NON] for exclusive non-target suppression.
        segmentation_tokens = ['[SEG]']
        non_tokens = ["[NON]"]
        # Adding tokens for GCG
        phrase_tokens = ['<p>', '</p>']
        special_tokens = reg_tokens + segmentation_tokens + phrase_tokens + non_tokens
        tokenizer.add_tokens(special_tokens, special_tokens=True)
    non_tokens = ["[NON]"]
    tokenizer.add_tokens(non_tokens, special_tokens=True)
    args.bbox_token_idx = tokenizer("<bbox>", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.bop_token_idx = tokenizer("<p>", add_special_tokens=False).input_ids[0]
    args.eop_token_idx = tokenizer("</p>", add_special_tokens=False).input_ids[0]
    args.non_token_idx = tokenizer("[NON]", add_special_tokens=False).input_ids[0]
    return tokenizer


def initialize_model(args, tokenizer):
    """Initialize the PestScope-compatible model."""
    model_args = {k: getattr(args, k) for k in
                  ["train_mask_decoder", "out_dim", "ce_loss_weight", "dice_loss_weight", "bce_loss_weight",
                   "seg_token_idx", "vision_pretrained", "vision_tower", "use_mm_start_end", "mm_vision_select_layer",
                   "pretrain_mm_mlp_adapter", "tune_mm_mlp_adapter", "freeze_mm_mlp_adapter", "mm_use_im_start_end",
                   "with_region", "bbox_token_idx", "eop_token_idx", "bop_token_idx","non_token_idx"]}
    model_args["num_level_reg_features"] = 4

    model = GLaMMForCausalLM.from_pretrained(
        args.version, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **model_args
    )
    print('\033[92m' + "---- Initialized model from: {} ----".format(args.version) + '\033[0m')

    # Configure model tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model


def prepare_model_for_training(model, tokenizer, args):
    # Enable input gradients
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Initialize vision tower
    print(
        '\033[92m' + "---- Initialized Global Image Encoder (vision tower) from: {} ----".format(
            args.vision_tower
        ) + '\033[0m'
    )
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=args.local_rank)

    # Initialize PestScope-compatible model and adjust requires_grad
    if not args.pretrained:
        model.get_model().initialize_glamm_model(model.get_model().config)
    else:
        for param in model.get_model().grounding_encoder.parameters():
            param.requires_grad = False
        if model.get_model().config.train_mask_decoder:
            model.get_model().grounding_encoder.mask_decoder.train()
            for param in model.get_model().grounding_encoder.mask_decoder.parameters():
                param.requires_grad = True

        # Projection layer
        model.get_model().text_hidden_fcs.train()
        for param in model.get_model().text_hidden_fcs.parameters():
            param.requires_grad = True

    # Set requires_grad for vision tower and mm projector
    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    # Set requires_grad based on LoRA training
    lora_r = args.lora_r
    if lora_r == 0:
        for p in model.get_model().layers.parameters():
            p.requires_grad = True
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    # Configure conversation library
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # Configure LoRA if applicable
    if lora_r > 0:
        lora_config = setup_lora_config(model, args)
        model = get_peft_model(model, lora_config)

    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Make certain modules trainable
    set_trainable_modules(model)  #


def setup_lora_config(model, args):
    """ Configure LoRA settings for the model. """

    def find_proj_layers(model, target_modules):
        """ Identify projection layers in the model for LoRA adaptation. """
        linear_cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if (isinstance(module, linear_cls) and all(
                    x not in name for x in ["grounding_encoder", "vision_tower", "mm_projector", "text_hidden_fcs"]
            ) and any(x in name for x in target_modules)):
                lora_module_names.add(name)
        return sorted(list(lora_module_names))

    # Extracting LoRA target modules
    lora_target_modules = args.lora_target_modules.split(",")
    lora_module_names = find_proj_layers(model, lora_target_modules)

    # Configuring LoRA
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=lora_module_names, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM"
    )
    return lora_config


def set_trainable_modules(model):
    """ Make specified modules in the model trainable. """
    trainable_modules = ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs", "region_encoder"]
    for name, param in model.named_parameters():
        if any(module in name for module in trainable_modules):
            print(f"Making trainable: {name}, Shape: {param.shape}")
            param.requires_grad = True

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('\033[92m' + "---- Total parameters: ----{}".format(total_params) + '\033[0m')
        print('\033[92m' + "---- Trainable parameters: ----{}".format(trainable_params) + '\033[0m')

    count_parameters(model)


def initialize_datasets_and_loaders(args, tokenizer):
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    # Common dataset arguments
    common_ds_args = {"dataset_dir": args.dataset_dir, "tokenizer": tokenizer,
                      "global_image_encoder": args.vision_tower,
                      "epoch_samples": args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size,
                      "precision": args.precision, "image_size": args.image_size,
                      "num_classes_per_sample": args.num_classes_per_sample}

    cap_dataset_classes = {"CocoCap": CocoCapDataset,
                           "LLaVaInstruct": LLaVAInstructDataset,
                           }
    reg_dataset_classes = {"RefCoco_Reg": RefCocoRegDataset,
                           "RefCocoG_Reg": RefCocoGRegDataset,
                           "RefCocoP_Reg": RefCocoPRegDataset,
                           "VisGen_Reg": VisualGenomeRegDataset,
                           "Flickr_Reg": Flickr30kRegDataset,
                           }
    seg_dataset_classes = {"Semantic_Segm": SemanticSegmDataset,
                           "Refer_Segm": ReferSegmDataset,
                           "PSG_GCG": OpenPsgGCGDataset,
                           "RefCoco_GCG": RefCOCOgGCGDataset,
                           "GranDf_GCG": GranDfDataset,
                           "Flickr_GCG": Flickr30kGCGDataset,
                          }
    # Train datasets
    if args.use_cap_data:
        train_datasets = [cap_dataset_classes[ds_name](**common_ds_args, random_sampling=False)
                          for ds_name in args.cap_dataset.split("||")]
    elif args.use_reg_data:
        train_datasets = [reg_dataset_classes[ds_name](**common_ds_args, random_sampling=False)
                          for ds_name in args.reg_dataset.split("||")]
    elif args.use_segm_data:
        train_datasets = []
        for ds_name in args.seg_dataset.split('||'):
            seg_dataset_class = seg_dataset_classes.get(ds_name)
            if seg_dataset_class:
                if seg_dataset_class == ReferSegmDataset:
                    all_datasets = args.refer_segm_data.split("||")
                    for d in all_datasets:
                        dataset_class = seg_dataset_class(**common_ds_args, random_sampling=False, refer_segm_data=d,
                        image_root_path=args.image_root_path,reason_seg_data_add=args.reason_seg_data_add,
                        reason_image_root_path=args.reason_image_root_path)
                        dataset_class._set_len(len(dataset_class.refer_segm_data[0]))


                        train_datasets.append(dataset_class)



                elif seg_dataset_class == SemanticSegmDataset:
                    all_datasets = args.semantic_segm_data.split("||")
                    for d in all_datasets:
                        dataset_class = seg_dataset_class(**common_ds_args, random_sampling=False, refer_segm_data=d)
                        dataset_class._set_len(len(dataset_class.semantic_segm_data[d]['images']))
                        train_datasets.append(dataset_class)
                else:
                    train_datasets.append(seg_dataset_class(**common_ds_args))
    else:

        train_datasets = []

    # Assert that exactly one dataset type is set
    dataset_types_set = sum([args.use_cap_data, args.use_reg_data, args.use_segm_data])
    assert dataset_types_set == 1, "Exactly one dataset type must be set"

    world_size = torch.cuda.device_count()
    # Summing lengths of all datasets
    total_length = sum(len(dataset) for dataset in train_datasets)
    print(f"Training with {total_length} examples.")
    # Calculate steps per epoch
    effective_batch_size = args.batch_size * args.grad_accumulation_steps * world_size
    steps_per_epoch = total_length // effective_batch_size
    # modify steps per epoch
    args.steps_per_epoch = steps_per_epoch

    # Concatenating datasets
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    # Validation datasets
    val_datasets = []
    if not args.no_eval:
        val_dataset_classes = {'CocoCapVal': CocoCapDataset,
                               'RefCOCOgRegVal': RefCocoGRegDataset,
                               'VisGenomeRegVal': VisualGenomeRegDataset,
                               'RefCOCOgSegmVal': ReferSegmDatasetVal,
                               'PsgGCGVal': OpenPsgGCGDataset,
                               'RefCocoGCGVal': RefCOCOgGCGDataset,
                               'FlickrGCGVal': Flickr30kGCGDataset,
                               }
        for val_dataset_name in args.val_dataset.split('|'):
            val_dataset_class = val_dataset_classes.get(val_dataset_name)
            if val_dataset_class:
                if val_dataset_class == ReferSegmDatasetVal:
                    # Modify this if other datasets in refer_segm_data need to be included in val
                    refer_segm_data = 'refcocog'
                    all_datasets = refer_segm_data.split("||")
                    for d in all_datasets:
                        val_dataset_class = val_dataset_class(
                            **common_ds_args, validation=True, refer_segm_data=d, split='val',image_root_path=args.image_root_path,reason_seg_data_add=args.reason_seg_data_add,
                        reason_image_root_path=args.reason_image_root_path)
                        val_dataset_class._set_len(len(val_dataset_class.images))
                        val_datasets.append(val_dataset_class)
                else:
                    val_datasets.append(val_dataset_class(**common_ds_args, validation=True))

    return train_dataset, val_datasets


def setup_data_loaders(args, train_dataset, val_datasets, tokenizer):
    sampler_args = {"shuffle": False, "drop_last": False}
    train_loader_args = {"batch_size": args.batch_size, "shuffle": False, "num_workers": args.workers,
                         "pin_memory": False}
    val_loader_args = {"batch_size": args.val_batch_size, "shuffle": False, "num_workers": args.workers,
                       "pin_memory": False}
    collate_fn_args_train = partial(
        custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank,
        inference=False
    )
    inference_mode = args.mask_validation
    collate_fn_args_val = partial(
        custom_collate_fn_val, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank,
        inference=inference_mode
    )

    # Training loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            train_dataset, **sampler_args
            ), collate_fn=collate_fn_args_train, **train_loader_args
        )

    # Validation loader
    val_loader = None
    if val_datasets:
        combined_val_datasets = ConcatDataset(val_datasets)
        val_loader = torch.utils.data.DataLoader(
            combined_val_datasets, **val_loader_args, collate_fn=collate_fn_args_val,
            sampler=torch.utils.data.distributed.DistributedSampler(combined_val_datasets, **sampler_args), )

    return train_loader, val_loader


def initialize_deepspeed(model, tokenizer, args):
    ds_config = {"train_micro_batch_size_per_gpu": args.batch_size,
                 "gradient_accumulation_steps": args.grad_accumulation_steps,
                 "optimizer": {"type": "AdamW", "params": {"lr": args.lr, "weight_decay": 0.0,
                                                           "betas": (args.beta1, args.beta2)}},
                 "scheduler": {"type": "WarmupDecayLR",
                               "params": {"total_num_steps": args.epochs * args.steps_per_epoch, "warmup_min_lr": 0,
                                          "warmup_max_lr": args.lr, "warmup_num_steps": 100, "warmup_type": "linear"}},
                 "fp16": {"enabled": args.precision == "fp16"}, "bf16": {"enabled": args.precision == "bf16"},
                 "gradient_clipping": 1.0,
                 "zero_optimization": {"stage": 2, "contiguous_gradients": True, "overlap_comm": True,
                                       "reduce_scatter": True, "reduce_bucket_size": 5e8,
                                       "allgather_bucket_size": 5e8}, }

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), collate_fn=partial(
            custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank
        ), config=ds_config
    )

    return model_engine, optimizer, scheduler

def resume_training_from_checkpoint(model_engine, args):
    if args.auto_resume and not args.resume:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        print(f"Resume training from {args.resume}, start from epoch {args.start_epoch}")


def main(args):
    tokenizer = setup_tokenizer_and_special_tokens(args)
    model = initialize_model(args, tokenizer)
    prepare_model_for_training(model, tokenizer, args)


    train_dataset, val_datasets = initialize_datasets_and_loaders(args, tokenizer)

    model_engine, optimizer, scheduler = initialize_deepspeed(model, tokenizer, args)
    resume_training_from_checkpoint(model_engine, args)

    train_loader, val_loader = setup_data_loaders(args, train_dataset, val_datasets, tokenizer)
    dataset_iter = iter(train_loader)

    writer = initialize_environment(args)

    if args.eval_only:
        random.seed(42)
        giou, ciou = validate_model_performance(val_loader, model_engine, "888", writer, args, tokenizer)
        exit()

    epoch_seeds = [random.randint(0, 100000) for _ in range(args.epochs)]

    giou, ciou = validate_model_performance(val_loader, model_engine, "888", writer, args, tokenizer)

    best_giou, best_ciou, best_val_loss = 0.0, 0.0, np.inf
    for epoch in range(args.start_epoch, args.epochs):
        random.seed(epoch_seeds[epoch])
        
        dataset_iter = train(train_loader, model_engine, epoch, scheduler, writer, dataset_iter, args)
        giou, ciou = validate_model_performance(val_loader, model_engine, epoch, writer, args, tokenizer)

        if epoch == 5:
            save_dir = os.path.join(args.log_dir, f"ckpt_model_{epoch}")
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)


def save_checkpoint(model_engine, args, epoch, metric_name, metric_value, is_best):
    """ Saves the model checkpoint. """
    # If the checkpoint is the best, save it in ckpt_model_best, else in ckpt_model_last_epoch
    save_dir_name = "ckpt_model_best" if is_best else "ckpt_model_last_epoch"
    save_dir = os.path.join(args.log_dir, save_dir_name)
    # Ensure the directory exists
    if args.local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        ckpt_filename = f"epoch_{epoch}_val_{metric_name}_{metric_value}.pth"
        torch.save({"epoch": epoch, f"val_{metric_name}": metric_value}, os.path.join(save_dir, ckpt_filename))
    torch.distributed.barrier()
    model_engine.save_checkpoint(save_dir)


def train(data_loader, model, epoch, scheduler, writer, dataset_iter, args):
    """Main training loop."""

    def get_next_input(iterator, data_loader):
        """Retrieve next input from the iterator, or reinitialize if necessary."""
        try:
            return next(iterator), iterator
        except StopIteration:
            new_iterator = iter(data_loader)
            return next(new_iterator), new_iterator

    def log_progress():
        """Log training progress."""
        if global_step % args.print_freq == 0:
            if args.distributed:
                for tracker in trackers.values():
                    tracker.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                for key, tracker in trackers.items():
                    writer.add_scalar(f"train/{key}", tracker.avg, global_step)
                writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, global_step)
                writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, global_step)

            for tracker in trackers.values():
                tracker.reset()

    batch_time = AverageMeter("Time", ":.4f")
    data_time = AverageMeter("Data", ":.4f")
    trackers = {"loss": AverageMeter("Loss", ":.4f"),
                "ce_loss": AverageMeter("CeLoss", ":.4f"),
                "mask_bce_loss": AverageMeter("MaskBCELoss", ":.4f"),
                "mask_dice_loss": AverageMeter("MaskDICELoss", ":.4f"),
                "seg_non_loss": AverageMeter("SegNonLoss", ":.4f"),
                "mask_loss": AverageMeter("MaskLoss", ":.4f")}
    progress = ProgressMeter(args.steps_per_epoch, list(trackers.values()), prefix=f"Epoch: [{epoch}]")

    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for _ in range(args.grad_accumulation_steps):
            # Select data loader based on step choice
            data_batch, new_iter = get_next_input(dataset_iter, data_loader)
            dataset_iter = new_iter

            data_time.update(time.time() - end)
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                if data_batch[key] is not None:
                    data_batch[key] = data_batch[key].bfloat16()

            output_dict = model(**data_batch)

            # Update training metrics
            for key, tracker in trackers.items():
                if key in output_dict:
                    tracker.update(output_dict[key].item(), data_batch["global_enc_images"].size(0))

            model.backward(output_dict["loss"])
            model.step()

        batch_time.update(time.time() - end)
        end = time.time()
        log_progress()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return dataset_iter




def save_sample_times_simple(time_image_paths, time_e2e, time_llm, time_mask_decode, csv_dir="output_csv"):
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"sample_times_rank{rank}.csv")
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["time_image_paths", "time_e2e", "time_llm", "time_mask_decode"])
        writer.writerow([time_image_paths, time_e2e, time_llm, time_mask_decode])


def validate_model_performance(val_loader, model_engine, epoch, writer, args, tokenizer):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()
    pest3_dict = {"铜绿丽金龟": "32", "东方蝼蛄": "34", "棉铃虫": "6"}

    pest24_dict = {'稻飞虱': '1', '稻纵卷叶螟': '2', '二化螟': '3', '黏虫': '5', '棉铃虫': '6', '草地螟': '7',
                   '二点委夜蛾': '8', '斜纹夜蛾': '10', '甜菜夜蛾': '11', '茎蛀虫?': '12', '小壁虎?': '13',
                   '小菜蛾': '14', '': '15', '三叶草夜蛾': '16', '黄地老虎': '24', '小地老虎': '25', '八字地老虎': '28',
                   '大黑鳃金龟': '29', '暗黑鳃金龟': '31', '铜绿丽金龟': '32', '东方蝼蛄': '34', '线虫': '35',
                   '金针虫': '36', '麦蛾': '37'}

    count_pest = {}
    sum_pest = {}

    pest_intersection_meter = {}
    pest_union_meter = {}
    pest_acc_iou_meter = {}

    for k, y in pest24_dict.items():
        count_pest[f"count_pest_{y}"] = AverageMeter(f"count_pest_{y}", ":6.3f", Summary.SUM)
        sum_pest[f"sum_pest_{y}"] = AverageMeter(f"sum_pest_{y}", ":6.3f", Summary.SUM)
        pest_intersection_meter[f"pest_intersection_meter_{y}"] = AverageMeter(f"pest_intersection_meter_{y}", ":6.3f", Summary.SUM)
        pest_union_meter[f"pest_union_meter_{y}"] = AverageMeter(f"pest_union_meter_{y}", ":6.3f", Summary.SUM)
        pest_acc_iou_meter[f"pest_acc_iou_meter_{y}"] = AverageMeter(f"pest_acc_iou_meter_{y}", ":6.3f", Summary.SUM)

    test_count = 0
    for input_dict in tqdm.tqdm(val_loader):
        test_count += 1
        if test_count == 100000:
            break
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["grounding_enc_images"] = input_dict["grounding_enc_images"].half()
            input_dict["global_enc_images"] = input_dict["global_enc_images"].half()
        elif args.precision == "bf16":
            input_dict["grounding_enc_images"] = input_dict["grounding_enc_images"].bfloat16()
            input_dict["global_enc_images"] = input_dict["global_enc_images"].bfloat16()
        else:
            input_dict["grounding_enc_images"] = input_dict["grounding_enc_images"].float()
            input_dict["global_enc_images"] = input_dict["global_enc_images"].float()

        with torch.no_grad():
            output_dict = model_engine(**input_dict)
        all_pests_list = input_dict["all_pest_list"]
        all_bboxes_list = input_dict["all_bbox_list"]
        all_masks_list = input_dict["all_mask_list"]
        bboxes_list = input_dict["bbox_list"]
        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()

        time_image_paths = input_dict["image_paths"]
        time_e2e = output_dict["time_e2e"]
        time_llm = output_dict["time_llm"]
        time_mask_decode = output_dict["time_mask_decode"]
        save_sample_times_simple(time_image_paths, time_e2e, time_llm, time_mask_decode)


        output_list = torch.max(output_list, dim=0)[0].unsqueeze(0)
        assert len(pred_masks) == 1

        image_path = input_dict["image_paths"][0]
        conversation = input_dict["conversation_list"][0]
        output_ids = output_dict["output_ids"]
        non_pred_masks = output_dict["non_pred_masks"][0]

        def get_output_image(image_path, conversation, pred_masks, gt_mask, non_pred_masks, output_ids, tokenizer,
                             save_dir="output_vis_refer", csv_path="output_vis_refer/records.csv"):

            os.makedirs(save_dir, exist_ok=True)
            image_name = image_path.split("/")[-1].split(".")[0]

            def save_mask(mask_tensor, save_path):
                if mask_tensor.shape[0] > 1:
                    mask_tensor = torch.max(mask_tensor, dim=0)[0]
                else:
                    mask_tensor = mask_tensor.squeeze(0)
                mask_tensor = mask_tensor.detach().cpu().numpy()
                mask_image = (mask_tensor * 255).astype(np.uint8)
                cv2.imwrite(save_path, mask_image)

            pred_mask = (pred_masks[0] > 0).int()

            pred_mask_name = f"{image_name}_pred_mask.jpg"
            gt_mask_name = f"{image_name}_gt_mask.jpg"

            save_mask(pred_mask, os.path.join(save_dir, pred_mask_name))
            save_mask(gt_mask, os.path.join(save_dir, gt_mask_name))

            if non_pred_masks.shape[0] > 0:
                merged_non_pred_mask = non_pred_masks > 0
                non_pred_masks_name = f"{image_name}_non_pred_mask_merged.jpg"
                save_mask(merged_non_pred_mask, os.path.join(save_dir, non_pred_masks_name))
            else:
                non_pred_masks_name = ""

            output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
            text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
            text_output = text_output.replace("\n", "").replace("  ", " ")

            header = ["image_path", "conversation", "pred_mask_name", "gt_mask_name", "non_pred_masks_name",
                      "text_output"]
            row = [image_path, conversation, pred_mask_name, gt_mask_name, non_pred_masks_name, text_output]

            write_header = not os.path.exists(csv_path)

            with open(csv_path, "a", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(header)
                writer.writerow(row)

            return text_output


        def get_com_mask(masks_list):
            merged_mask = torch.max(masks_list, dim=0)[0]
            merged_mask = merged_mask.unsqueeze(0).to(masks_list.device)
            return merged_mask

        if masks_list.size(0) != 1:
            merged_mask = get_com_mask(masks_list)
            masks_list = merged_mask
        intersection, union, acc_iou = 0.0, 0.0, 0.0


        for mask_i, output_i, bbox_list, all_bbox_list, all_pest_list, all_mask_list in zip(masks_list, output_list,
                                                                                            bboxes_list,
                                                                                            all_bboxes_list,
                                                                                            all_pests_list,
                                                                                            all_masks_list):

            tt_list = []
            for m in all_mask_list:
                m = torch.tensor(m, dtype=torch.int32)
                m = m.to(output_i.device)
                tt_list.append(m)
            all_mask_list = tt_list
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0  # no-object target


            labels = []
            for i, item in enumerate(all_pest_list):
                label = 1 if item in input_dict["text_list"][0] else 0
                if len(all_bbox_list[i]) > 1:
                    labels.append([label] * len(all_bbox_list[i]))
                else:
                    labels.append([label])

            for pest_n, bboxes_s in zip(input_dict["text_list"][0], bboxes_list[0]):
                pest_id = pest24_dict[pest_n]
                sum_pest_id = f"sum_pest_{pest_id}"
                p_num = len(bboxes_s)
                if sum_pest_id in sum_pest:
                    sum_pest[sum_pest_id].update(1, n=p_num)
                    # pest_id_gt = pest_id

            bbox_iou_list = []
            is_more_pred = False
            for i, label_list in enumerate(labels):
                bboxes = all_bbox_list[i]
                pest_name = all_pest_list[i]
                pest_name_id = pest24_dict[pest_name]
                mask = all_mask_list[i]

                bbox_iou_list_tmp = []
                pest_intersection, pest_union, pest_acc_iou = 0.0, 0.0, 0.0
                for lbl, bbox in zip(label_list, bboxes):
                    x_min, y_min, x_max, y_max = bbox
                    x_min, x_max = max(0, x_min), min(output_i.shape[1], x_max)
                    y_min, y_max = max(0, y_min), min(output_i.shape[0], y_max)
                    mask_crop = mask[y_min:y_max, x_min:x_max]
                    output_crop = output_i[y_min:y_max, x_min:x_max]
                    bbox_intersection, bbox_union, _ = intersectionAndUnionGPU(
                        output_crop.contiguous().clone(),
                        mask_crop.contiguous(),
                        2,
                        ignore_index=255
                    )
                    bbox_iou = bbox_intersection / (bbox_union + 1e-5)
                    bbox_iou = bbox_iou.cpu().numpy()
                    bbox_iou_list_tmp.append(bbox_iou[1])

                    if lbl == 1:
                        pest_intersection += bbox_intersection
                        pest_union += bbox_union
                        pest_acc_iou += bbox_intersection / (bbox_union + 1e-5)
                        pest_acc_iou[bbox_union == 0] += 1.0  # no-object target
                    if lbl == 0:
                        # if bbox_union[1] > 100:
                        if bbox_iou[1] > 0.2:
                            is_more_pred = True

                if label_list[0] == 1:
                    pest_intersection, pest_union = pest_intersection.cpu().numpy(), pest_union.cpu().numpy()
                    pest_acc_iou = pest_acc_iou.cpu().numpy() / len(label_list)

                    pest_intersection_meter_id = f"pest_intersection_meter_{pest_name_id}"
                    pest_union_meter_id = f"pest_union_meter_{pest_name_id}"
                    pest_acc_iou_meter_id = f"pest_acc_iou_meter_{pest_name_id}"
                    pest_intersection_meter[pest_intersection_meter_id].update(pest_intersection)
                    pest_union_meter[pest_union_meter_id].update(pest_union)
                    pest_acc_iou_meter[pest_acc_iou_meter_id].update(pest_acc_iou, n=len(label_list))


                bbox_iou_list.append(bbox_iou_list_tmp)
            if not is_more_pred:
                for i, label_list in enumerate(labels):
                    bboxes = all_bbox_list[i]
                    pest_name = all_pest_list[i]
                    mask = all_mask_list[i]
                    pest_id = pest24_dict[pest_name]
                    for j, lbl in enumerate(label_list):
                        iou = bbox_iou_list[i][j]
                        if lbl == 1:
                            # assert all_mask_list[i].equal(mask_i)
                            if iou > 0.5:
                                count_pest_id = f"count_pest_{pest_id}"
                                count_pest[count_pest_id].update(1, n=1)


        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]

        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    pest_iou_class = {}
    pest_ciou = {}
    pest_giou = {}
    for k1, y1 in pest24_dict.items():
        count_pest[f"count_pest_{y1}"].all_reduce()
        sum_pest[f"sum_pest_{y1}"].all_reduce()
        pest_intersection_meter[f"pest_intersection_meter_{y1}"].all_reduce()
        pest_union_meter[f"pest_union_meter_{y1}"].all_reduce()
        pest_acc_iou_meter[f"pest_acc_iou_meter_{y1}"].all_reduce()
        if sum_pest[f"sum_pest_{y1}"].count != 0:
            pest_iou_class[f"pest_iou_class_{y1}"] = pest_intersection_meter[f"pest_intersection_meter_{y1}"].sum / (pest_union_meter[f"pest_union_meter_{y1}"].sum + 1e-10)
            pest_ciou[f"pest_ciou_{y1}"] = pest_iou_class[f"pest_iou_class_{y1}"][1]
            pest_giou[f"pest_giou_{y1}"] = pest_acc_iou_meter[f"pest_acc_iou_meter_{y1}"].avg[1]


    if args.local_rank == 0:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

        output_path = "results.txt"
        with open(output_path, "a") as f:
            f.write(f"Epoch: {epoch}\n")
            f.write("giou: {:.4f}, ciou: {:.4f}\n".format(giou, ciou))
            av_acc = 0
            sum_acc = 0
            sum_pest_num = 0
            for n, i in pest24_dict.items():
                if sum_pest[f"sum_pest_{i}"].count != 0:
                    sum_pest_num += 1
                    #####
                    f.write("count_pest_{}: {:.4f}, sum_pest_{}: {:.4f}, acc_pest_{}: {:.4f},pest_giou_{}: {:.4f},pest_ciou_{}: {:.4f} \n".format(i,
                                                                                                       count_pest[
                                                                                                           f"count_pest_{i}"].count,
                                                                                                       i, sum_pest[
                                                                                                           f"sum_pest_{i}"].count,
                                                                                                       i, count_pest[
                                                                                                           f"count_pest_{i}"].count /
                                                                                                       sum_pest[
                                                                                                           f"sum_pest_{i}"].count,i,pest_giou[f"pest_giou_{i}"],i,pest_ciou[f"pest_ciou_{i}"]))

                    acc = count_pest[f"count_pest_{i}"].count / sum_pest[f"sum_pest_{i}"].count
                    sum_acc += acc
            f.write("av_acc: {:.4f}\n".format(sum_acc / sum_pest_num))

            f.write("\n")
            # av_acc_ = sum_acc / sum_pest_num
    return giou, ciou



if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
