# git clone https://github.com/huggingface/transformers.git
import sys
sys.path.append('transformers/examples/research_projects/lxmert/')
sys.path.append('transformers/src/')

import torch
import numpy as np
from processing_image import Preprocess
from visualizing_image import SingleImageViz
from modeling_frcnn import GeneralizedRCNN
from utils import Config
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
    def __init__(self, config='unc-nlp/frcnn-vg-finetuned'):
        self.frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
        self.frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=self.frcnn_cfg)
        self.image_preprocess = Preprocess(self.frcnn_cfg)

    def __call__(self, images, batch_size=100):
        images, sizes, scales_yx = self.image_preprocess(images)
        output_dict = self.frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            max_detections=self.frcnn_cfg.max_detections,
            return_tensors="pt"
        )
        #(batch_size, 36, 4)
        normalized_boxes = output_dict.get("normalized_boxes")
        #(batch_size, 36, 2048)
        features = output_dict.get("roi_features")
        return features, normalized_boxes
