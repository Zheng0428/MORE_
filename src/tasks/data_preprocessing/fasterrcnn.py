import sys
#sys.path.append("./src/tasks/data_preprocessing")
import torch
import numpy as np
from processing_image import Preprocess
#from visualizing_image import SingleImageViz
from modeling_frcnn import GeneralizedRCNN
from tasks.data_preprocessing.utils import Config, extend_tensor
import utils
import wget
import os

'''faster r-cnn visual encoder for LXMert model
Usage:

faster_r_cnn = FasterRCNN_Visual_Feats()
visual_feats, visual_pos = faster_r_cnn(images)
#visual_feats -> (batch_size, 36, 2048)
#visual_pos -> (batch_size, 36, 4)

# inputs from lxmert tokenizer
model = LxmertModel.from_pretrained(config)
outputs = model(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    visual_feats=visual_feats,
    visual_pos=visual_pos,
    token_type_ids=inputs.token_type_ids,
    return_dict=True,
    output_attentions=False,
)

TO DO:

make compatible with GPU (only works with cpu right now) to make workable for traj dataset
'''

class FasterRCNN_Visual_Feats:
    def __init__(self, device, config='unc-nlp/frcnn-vg-finetuned'):
        self.device = device
        self.frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn_cfg.model.device = self.device
        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg).to(self.device)
        self.image_preprocess = Preprocess(self.frcnn_cfg)

    def __call__(self, images):
        features, normalized_boxes= None, None
        for img in images:
            images, sizes, scales_yx = self.image_preprocess([img])
            output_dict = self.frcnn(
                images.to(self.device),
                sizes.to(self.device),
                scales_yx=scales_yx.to(self.device),
                padding="max_detections",
                max_detections=self.frcnn_cfg.max_detections,
                return_tensors="pt"
                )
            #(batch_size, 36, 4)
            normalized_boxes_batch = output_dict.get("normalized_boxes")
            #(batch_size, 36, 2048)
            features_batch = output_dict.get("roi_features")
            features = extend_tensor(features, features_batch)
            normalized_boxes = extend_tensor(normalized_boxes, normalized_boxes_batch)
        return features, normalized_boxes
