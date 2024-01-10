import clip
import torch
import utils
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _preprocess_manual(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        #ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


#a function that takes parametrs of gaussians rander a 2d image and then use clip image encoder to encode it
def clip_image_encoder(image, device):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    preprocess = _preprocess_manual(model.visual.input_resolution)
    image = preprocess(image).unsqueeze(0).to(device) # the unsqueeze is needed to add a batch dimension

    with torch.no_grad():
        image_features = model.encode_image(image) # size of [1,512] for ViT-B/32

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features



