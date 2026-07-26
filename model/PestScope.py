import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F
import time
from model.SAM import build_sam_vit_h
from model.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaLlamaModel


def calculate_dice_loss(predictions: torch.Tensor, ground_truth: torch.Tensor, mask_count: float, scale_factor=1000,
                        epsilon=1e-6):
    """
    Calculate the DICE loss, a measure similar to generalized IOU for masks.
    """
    predictions = predictions.sigmoid()
    predictions = predictions.flatten(1, 2)
    ground_truth = ground_truth.flatten(1, 2)

    intersection = 2 * (predictions / scale_factor * ground_truth).sum(dim=-1)
    union = (predictions / scale_factor).sum(dim=-1) + (ground_truth / scale_factor).sum(dim=-1)

    dice_loss = 1 - (intersection + epsilon) / (union + epsilon)
    dice_loss = dice_loss.sum() / (mask_count + 1e-8)
    return dice_loss


def compute_sigmoid_cross_entropy(predictions: torch.Tensor, targets: torch.Tensor, mask_count: float):
    """
    Compute sigmoid cross-entropy loss for binary classification.
    """
    loss = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1)
    loss = loss.sum() / (mask_count + 1e-8)
    return loss

def overlap_loss(inputs: torch.Tensor,
                 targets: torch.Tensor,
                 num_masks: float):
    if num_masks == 0:
        return inputs.sum() * 0

    loss = 0
    question_inputs = inputs
    question_targets = targets

    n, h, w = question_inputs.shape  


    overlap_area = (question_inputs > 0).sum(dim=0)
    overlap_area = overlap_area >= 1

    overlap_area = overlap_area[None].repeat(n, 1, 1)
    weight = torch.ones_like(question_inputs)
    weight[~overlap_area] = 0

    q_loss = F.binary_cross_entropy_with_logits(question_inputs, question_targets, weight=weight, reduction="none")
    q_loss = q_loss.flatten(1, 2).mean(1).sum()
    loss = loss + q_loss
    return loss

def non_loss(pred_mask: torch.Tensor,
                 targets_mask: torch.Tensor,
                 num_masks: float,
                 bool_tensor_):
    if num_masks == 0:
        return pred_mask.sum() * 0
    device = pred_mask.device
    assert bool_tensor_.shape[0] == pred_mask.shape[0]

    seg_mask = pred_mask[bool_tensor_]
    non_mask = pred_mask[~bool_tensor_]

    seg_n, seg_h, seg_w = seg_mask.shape
    non_n, non_h, non_w = non_mask.shape

    seg_targets_area = (seg_mask > 0).sum(dim=0)
    non_targets_area = (non_mask > 0).sum(dim=0)
    seg_targets_area = seg_targets_area >= 1
    non_targets_area = non_targets_area >= 1

    seg_targets_area = seg_targets_area[None].repeat(seg_n, 1, 1)
    non_targets_area = non_targets_area[None].repeat(non_n, 1, 1)

    seg_weight = torch.where(seg_targets_area, torch.tensor(4.0).to(device), torch.tensor(0.0).to(device))
    non_weight = torch.where(non_targets_area, torch.tensor(1.0).to(device), torch.tensor(0.0).to(device))

    # weight = torch.zeros_like(pred_mask)
    # weight[bool_tensor_] = seg_weight
    # weight[~bool_tensor_] = non_weight

    weight = torch.cat((seg_weight, non_weight), dim=0)

    loss = F.binary_cross_entropy_with_logits(pred_mask, targets_mask, weight=weight, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum()
    loss = loss / (num_masks + 1e-8)
    return loss
class GLaMMBaseModel:
    def __init__(self, config, **kwargs):
        super(GLaMMBaseModel, self).__init__(config)
        self.config = config
        self.vision_pretrained = kwargs.get("vision_pretrained", None)

        # Set config attributes if they don't exist
        self.config.train_mask_decoder = getattr(
            self.config, "train_mask_decoder", kwargs.get("train_mask_decoder", False)
        )
        self.config.out_dim = getattr(self.config, "out_dim", kwargs.get("out_dim", 512))

        self.initialize_glamm_model(self.config)

    def initialize_glamm_model(self, config):
        # Initialize the visual model
        self.grounding_encoder = build_sam_vit_h(self.vision_pretrained)
        self._configure_grounding_encoder(config)

        # Initialize the text projection layer
        self._initialize_text_projection_layer()

    def _configure_grounding_encoder(self, config):
        # Freezing visual model parameters
        for param in self.grounding_encoder.parameters():
            param.requires_grad = False

        # Training mask decoder if specified
        if config.train_mask_decoder:
            self._train_mask_decoder()

    def _train_mask_decoder(self):
        self.grounding_encoder.mask_decoder.train()
        for param in self.grounding_encoder.mask_decoder.parameters():
            param.requires_grad = True

    def _initialize_text_projection_layer(self):
        in_dim, out_dim = self.config.hidden_size, self.config.out_dim
        text_projection_layers = [nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0), ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_projection_layers)])
        self.text_hidden_fcs.train()
        self.text_hidden_fcs.train()


class GLaMMModel(GLaMMBaseModel, LlavaLlamaModel):
    def __init__(self, config, **kwargs):
        super(GLaMMModel, self).__init__(config, **kwargs)
        self._configure_model_settings()

    def _configure_model_settings(self):
        self.config.use_cache = False
        self.config.vision_module = self.config.mm_vision_module
        self.config.select_feature_type = "patch"
        self.config.image_aspect = "square"
        self.config.image_grid_points = None
        self.config.tune_mlp_adapter = False
        self.config.freeze_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.use_image_patch_token = False


class GLaMMForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, **kwargs):
        self._set_model_configurations(config, kwargs)
        super().__init__(config)
        self.model = GLaMMModel(config, **kwargs)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _set_model_configurations(self, config, kwargs):
        config.mm_use_image_start_end = kwargs.pop("use_mm_start_end", True)
        config.mm_vision_module = kwargs.get("vision_module", "openai/clip-vit-large-patch14-336")
        self._initialize_loss_weights(kwargs)
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 1)
        config.num_reg_features = kwargs.get("num_level_reg_features", 4)
        config.with_region = kwargs.get("with_region", True)
        config.bbox_token_idx = kwargs.get("bbox_token_idx", 32002)
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.non_token_idx = kwargs.pop("non_token_idx")

    def _initialize_loss_weights(self, kwargs):
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
        self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
        self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        self.seg_non_loss_weight = self.bce_loss_weight
    def get_grounding_encoder_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            return torch.cat([self._encode_single_image(img) for img in pixel_values], dim=0)

    def _encode_single_image(self, image):
        torch.cuda.empty_cache()
        return self.model.grounding_encoder.image_encoder(image.unsqueeze(0))

    def forward(self, **kwargs):
        return super().forward(**kwargs) if "past_key_values" in kwargs else self.model_forward(**kwargs)

    def model_forward(self, global_enc_images: torch.FloatTensor, grounding_enc_images: torch.FloatTensor,
                      bboxes: torch.FloatTensor, input_ids: torch.LongTensor, labels: torch.LongTensor,
                      attention_masks: torch.LongTensor, offset: torch.LongTensor, masks_list: List[torch.FloatTensor],
                      label_list: List[torch.Tensor], resize_list: List[tuple], inference: bool = False, **kwargs, ):
        non_masks_list = kwargs.get("non_masks_list", None)

        if inference:

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start_e2e = time.time()


            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_llm_start = time.time()

        if not non_masks_list:
            print("kwargs",kwargs)

        # Handle inference or training paths
        if inference:
            output_hidden_states, output_ids = self._inference_path(input_ids, global_enc_images, attention_masks)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_llm_end = time.time()
            t_llm = t_llm_end - t_llm_start
            # print(f"[TIME] LLM token generation: {t_llm:.4f}s")

        else:
            output, output_hidden_states = self._training_path(
                global_enc_images, bboxes, input_ids, labels, attention_masks, offset
            )


        if grounding_enc_images is not None:
            if inference:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_mask_start = time.time()

            # Extract grounding encoder image embeddings
            image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
            assert image_embeddings.shape[0] == len(offset) - 1

            # Create segmentation token mask
            seg_token_mask,non_token_mask = self._create_seg_token_mask(input_ids)


            # Process hidden states
            hidden_states, pred_embeddings,seg_token_counts, non_embeddings, non_token_counts = self._process_hidden_states(output_hidden_states, seg_token_mask, offset,non_token_mask)



            # Generate and post-process masks
            pred_masks = self._generate_and_postprocess_masks(
                pred_embeddings, image_embeddings, resize_list, label_list
            )

            if inference:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_mask_end = time.time()
                t_mask_decode = t_mask_end - t_mask_start
                # print(f"[TIME] mask decode: {t_mask_decode:.4f}s")


            train_gt_masks = []
            for seg_mask, non_mask, non_token_count in zip(masks_list, non_masks_list, non_token_counts):
                if non_token_count.item() != 0:
                    if non_mask.size(0) != 0:
                        train_gt_masks.append(torch.cat([seg_mask, non_mask], dim=0))
                    else:
                        non_mask = torch.zeros_like(seg_mask)
                        non_mask = non_mask[:1, ...]
                        train_gt_masks.append(torch.cat([seg_mask, non_mask], dim=0))
                else:
                    train_gt_masks.append(seg_mask)
            gt_masks = train_gt_masks

            # 
            inference_pred_masks = []
            non_pred_masks = []; bool_tensor_list = []
            for i in range(len(pred_embeddings)):
                seg_num = seg_token_counts[i].item()
                non_num = non_token_counts[i].item()
                real_seg_num = seg_num - non_num
                # 
                bool_tensor = torch.cat(
                    [torch.ones(real_seg_num, dtype=torch.bool), torch.zeros(non_num, dtype=torch.bool)])
                inference_pred_masks.append(pred_masks[i][bool_tensor])
                non_pred_masks.append(pred_masks[i][~bool_tensor])
                bool_tensor_list.append(bool_tensor)

            if inference:
        
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_e2e_end = time.time()
                t_e2e = t_e2e_end - t_start_e2e
                # conversation_list = kwargs.get("conversation_list", None)
                return {"pred_masks": inference_pred_masks,
                        "gt_masks": masks_list,
                        "non_pred_masks": non_pred_masks,
                        "output_ids": output_ids,

                        "time_e2e": t_e2e,
                        "time_llm": t_llm,
                        "time_mask_decode": t_mask_decode,
                        }
        else:
            pred_masks = None

        # Calculate losses
        return self._calculate_losses(pred_masks, gt_masks, output,bool_tensor_list)

    def _create_seg_token_mask(self, input_ids):
        mask = (input_ids[:, 1:] == self.seg_token_idx) | (input_ids[:, 1:] == self.non_token_idx)
        non_token_mask = input_ids[:, 1:] == self.non_token_idx
        return torch.cat(
            [torch.zeros((mask.shape[0], 575)).bool().cuda(), mask, torch.zeros((mask.shape[0], 1)).bool().cuda()],
            dim=1
        ),torch.cat(
            [torch.zeros((non_token_mask.shape[0], 575)).bool().cuda(), non_token_mask, torch.zeros((non_token_mask.shape[0], 1)).bool().cuda()],
            dim=1
        )

    def _inference_path(self, input_ids, global_enc_images, attention_masks):
        length = input_ids.shape[0]
        global_enc_images_extended = global_enc_images.expand(length, -1, -1, -1).contiguous()

        # Process and return inference output
        output_hidden_states = []
        for i in range(input_ids.shape[0]):

            output_i = self.generate(
                images=global_enc_images_extended[i:i + 1],
                attention_mask=attention_masks[i:i + 1],
                input_ids=input_ids[i:i + 1],
                max_new_tokens=32,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_ids = output_i.sequences
            output_hidden_states.append(output_i.hidden_states[0])

            torch.cuda.empty_cache()

        output_hidden_states = torch.cat(output_hidden_states, dim=0)
        output_hidden_states = [output_hidden_states]
        return output_hidden_states, output_ids

    def _training_path(self, global_enc_images, bboxes, input_ids, labels, attention_masks, offset):
        global_enc_images = self._prepare_global_enc_image(global_enc_images, offset)
        bboxes_list = bboxes

        output = super().forward(
            images=global_enc_images, attention_mask=attention_masks, input_ids=input_ids, labels=labels,
            output_hidden_states=True, bboxes=bboxes_list, )
        output_hidden_states = output.hidden_states
        return output, output_hidden_states

    def _prepare_global_enc_image(self, global_enc_image, offset):
        global_enc_image_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            global_enc_image_i = global_enc_image[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous()
            global_enc_image_list.append(global_enc_image_i)
        return torch.cat(global_enc_image_list, dim=0)

    def _process_hidden_states(self, output_hidden_states, seg_token_mask, offset,non_token_mask, infer=False):
        hidden_states = [self.model.text_hidden_fcs[0](output_hidden_states[-1])]
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        combined_token_mask = seg_token_mask | non_token_mask
        pred_embeddings = last_hidden_state[combined_token_mask]

        non_embeddings = last_hidden_state[non_token_mask]
        #
        seg_token_counts = seg_token_mask.int().sum(-1)  # [bs, ]
        non_token_counts = non_token_mask.int().sum(-1)
        combined_token_counts = combined_token_mask.int().sum(-1)

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        non_token_offset = non_token_counts.cumsum(-1)
        non_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), non_token_offset], dim=0
        )

        combined_token_offset = combined_token_counts.cumsum(-1)
        combined_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), combined_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]
        non_token_offset = non_token_offset[offset]
        combined_token_offset = combined_token_offset[offset]

        pred_embeddings_ = []
        non_embeddings_ = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            start_i_non, end_i_non = non_token_offset[i], non_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
            non_embeddings_.append(non_embeddings[start_i_non:end_i_non])
        pred_embeddings_list = pred_embeddings_
        non_embeddings = non_embeddings_
        return hidden_states, pred_embeddings_list,seg_token_counts, non_embeddings,non_token_counts

    def _generate_and_postprocess_masks(self, pred_embeddings, image_embeddings, resize_list, label_list, infer=False):
        pred_masks = []
        for i, pred_embedding in enumerate(pred_embeddings):
            sparse_embeddings, dense_embeddings = self.model.grounding_encoder.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=pred_embedding.unsqueeze(1)
            )
            sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
            low_res_masks, _ = self.model.grounding_encoder.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.grounding_encoder.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                multimask_output=False, )
            orig_size = label_list[i].shape if not infer else label_list[i]
            # During inference, we have original size list in place of label list
            pred_mask = self.model.grounding_encoder.postprocess_masks(
                low_res_masks, input_size=resize_list[i], original_size=orig_size, )
            pred_masks.append(pred_mask[:, 0])
        return pred_masks

    def _calculate_losses(self, pred_masks, masks_list, output,bool_tensor_list):
        loss_components = self._compute_loss_components(pred_masks, masks_list, output, bool_tensor_list)
        return loss_components

    def _compute_loss_components(self, pred_masks, masks_list, output,bool_tensor_list):
        # Initialize loss components
        ce_loss = output.loss * self.ce_loss_weight
        mask_bce_loss = torch.tensor(0.0, device=ce_loss.device)
        mask_dice_loss = torch.tensor(0.0, device=ce_loss.device)
        seg_non_loss = torch.tensor(0.0, device=ce_loss.device)
        num_masks = 0

        if pred_masks:
            # Iterate over batch and compute mask-related losses
            for batch_idx, pred_mask in enumerate(pred_masks):
                if pred_mask.numel() > 0:  # Ensure pred_mask is not empty
                    gt_mask = masks_list[batch_idx]
                    bool_tensor_ = bool_tensor_list[batch_idx]
                    # Resize gt_mask to match pred_mask if needed
                    if gt_mask.shape[0] != pred_mask.shape[0]:
                        gt_mask = gt_mask[:pred_mask.shape[0]]

                    assert gt_mask.shape[0] == pred_mask.shape[
                        0], f"Shape mismatch: gt_mask {gt_mask.shape}, pred_mask {pred_mask.shape}"

                    # Compute Binary Cross-Entropy Loss
                    mask_bce_loss += (compute_sigmoid_cross_entropy(pred_mask, gt_mask, mask_count=gt_mask.shape[0]) *
                                      gt_mask.shape[0])
                    # Compute Dice Loss
                    mask_dice_loss += (
                            calculate_dice_loss(pred_mask, gt_mask, mask_count=gt_mask.shape[0]) * gt_mask.shape[0])

                    seg_non_loss += (non_loss(pred_mask, gt_mask, gt_mask.shape[0],bool_tensor_) * gt_mask.shape[0])

                    num_masks += gt_mask.shape[0]

        # Normalize the losses
        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        seg_non_loss = self.seg_non_loss_weight * seg_non_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss + seg_non_loss

        # Aggregate all loss components
        total_loss = ce_loss + mask_loss
        return {"loss": total_loss, "ce_loss": ce_loss, "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss, "mask_loss": mask_loss,"seg_non_loss": seg_non_loss}

    def evaluate(self, global_enc_images, grounding_enc_images, input_ids, resize_list, orig_sizes, max_tokens_new=32,
                 bboxes=None, ):
        with torch.no_grad():
            generation_outputs = self.generate(
                images=global_enc_images, input_ids=input_ids, bboxes=bboxes, max_new_tokens=max_tokens_new,
                num_beams=1, output_hidden_states=True, return_dict_in_generate=True, )

            output_hidden_states = generation_outputs.hidden_states
            generated_output_ids = generation_outputs.sequences

            seg_token_mask = generated_output_ids[:, 1:] == self.seg_token_idx
            # Adjusting for IMAGE_TOKEN_INDEX (assuming single image at start)
            seg_token_mask = torch.cat(
                [torch.zeros((seg_token_mask.shape[0], 575), dtype=torch.bool).cuda(), seg_token_mask], dim=1, )
            # Process hidden states
            hidden_states, predicted_embeddings,non_embeddings = self._process_hidden_states(
                output_hidden_states, seg_token_mask, None, infer=True
            )
            image_embeddings = self.get_grounding_encoder_embs(grounding_enc_images)
            # Generate and post-process masks
            pred_masks = self._generate_and_postprocess_masks(
                predicted_embeddings, image_embeddings, resize_list, orig_sizes, infer=True
            )
        return generated_output_ids, pred_masks
