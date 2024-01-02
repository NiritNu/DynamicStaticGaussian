import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
from train import params2rendervar, Renderer



# create a function to visualize an image tensor ans save it to a file    
def show_save_image(img_tensor, img_name):
    #check if tensor is on cpu or gpu
    if img_tensor.device.type == 'cpu':
        img_tensor = img_tensor.detach()
    else:
        img_tensor = img_tensor.cpu().detach()
    plt.imshow(img_tensor.permute(1, 2, 0))
    plt.show()      # display the image             
    plt.savefig(img_name)    # save the image to local file system


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid((images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images in dl:
        show_images(images, nmax)
        break

# create a function that track the differance over itrations for a given parameter
def track_diff(param):                              
    param_diff = []
    for i in range(len(param)-1):
        param_diff.append(torch.norm(param[i+1]-param[i]))
    return param_diff

#create a function that takes parametrs and renders them as images
def  render_param(params, curr_data, img_name):
    #loop over dictonary params and detach the tensors
    for key in params:
        params[key] = params[key].detach() 
    rendervar = params2rendervar(params)
    #rendervar['means2D'].retain_grad()
    im, _, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    curr_id = curr_data['id']
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
    show_save_image(im, img_name)

def delete_rows_by_thresh(arr, th):
    bool_tensor = arr[arr<th]
    new_arr = arr[bool_tensor]
    return new_arr

# go through params dictonary and delete rows of the vales that are below a certain threshold
def delete_rows(params, th):
    for key in params:
        if key != 'cam_m' | key != 'cam_c':
            params[key] = delete_rows_by_thresh(params[key], th)
    return params  