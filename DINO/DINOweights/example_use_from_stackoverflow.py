import torch
from torchvision import transforms as pth_transforms 
from PIL import Image 

model_dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)

model_dino.eval()

#if GPU is available
device = torch.device("cuda")
model_dino.to(device)
image_path = '/home/nirit/3D/DynamicStaticGaussian/hist2.png'#'/home/nirit/3D/DynamicStaticGaussian/image.png'

transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


img = Image.open(image_path).convert('RGB')
img_tensor = transform(img)

img_tensor = img_tensor.unsqueeze(0).cuda()
predictions = model_dino(img_tensor) 