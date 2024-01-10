from segment_anything import SamPredictor, sam_model_registry

from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn.functional as F


def extract_feature_from_sam(
        im_path: str,
        sam_model_name: str,
        chechpoint_path: str):
   
    pil_image = Image.open(image_path).convert('RGB')

    # Convert the PIL image to a numpy array
    numpy_image = np.array(pil_image)

    sam = sam_model_registry[sam_model_name](checkpoint=chechpoint_path)
    predictor = SamPredictor(sam)
    predictor.set_image(numpy_image)
    scale = predictor.scale

    #interpolate the features to the input size (sam.image_encoder.img_size, sam.image_encoder.img_size)
    fetuare_pre_pixel = F.interpolate(predictor.features, (sam.image_encoder.img_size, sam.image_encoder.img_size), mode="bilinear", align_corners=False)
    #undo the padding and then the scalling
    h,w = predictor.input_size
    fetuare_pre_pixel_no_pad= fetuare_pre_pixel[..., :h, :w]
    #scale down emb_no_pad by factor scale
    feature_to_original_image_size = F.interpolate(fetuare_pre_pixel_no_pad, (int((fetuare_pre_pixel_no_pad.shape[-2]+0.5)//scale), int((fetuare_pre_pixel_no_pad.shape[-1]+0.5)//scale)), mode="bilinear", align_corners=False)
    return feature_to_original_image_size




