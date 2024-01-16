from segment_anything import SamPredictor, sam_model_registry

from PIL import Image
import numpy as np
from torchvision import transforms
import torch.nn.functional as F


def feature_extracter(sam_model_name, checkpoint_path, image_path):

    sam = sam_model_registry[sam_model_name](checkpoint=checkpoint_path)
    predictor = SamPredictor(sam)

    pil_image = Image.open(image_path).convert('RGB')

    # Convert the PIL image to a numpy array
    numpy_image = np.array(pil_image)

    # Define the transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((sam.image_encoder.img_size, sam.image_encoder.img_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply the transformations
    tensor_image = transform(numpy_image)

    # Add an extra dimension for the batch size
    tensor_image = tensor_image.unsqueeze(0)

    # Now you can pass tensor_image to the set_image method of the SamPredictor class
    predictor = SamPredictor(sam)
    '''image (np.ndarray): The image for calculating masks.
    Expects an image in HWC uint8 format, with pixel values in [0, 255].
            image_format (str): The color format of the image, in ['RGB', 'BGR'].'''
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
