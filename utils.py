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
def  render_param(params, curr_data, img_name, save_im = False):
    #loop over dictonary params and detach the tensors
    for key in params:
        params[key] = params[key].detach() 
    rendervar = params2rendervar(params)
    #rendervar['means2D'].retain_grad()
    im, _, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    curr_id = curr_data['id']
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
    if save_im:
        show_save_image(im, img_name)
    return im

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

#calaulate speed from position difference and frame rate
def calc_speed(pos_diff):
    frame_rate = 15/0.5 # 0.5 seconds per 15 frames as mention in the paper web page
    speed = pos_diff*frame_rate
    return speed

# take all parameters and concat them to one tensor
def concat_params(params):
    concat_params = torch.cat((params['means3D'], params['colors_precomp'], params['rotations'], params['opacities'], params['scales'],params['means2D']), dim=1)
    return concat_params

#takes one tensor and split it to the different parameters the opposite of concat_params
def split_params(params):
    params_dict = {}
    params_dict['means3D'] = params[:, :3]
    params_dict['colors_precomp'] = params[:, 3:6]
    params_dict['rotations'] = params[:, 6:9]
    params_dict['opacities'] = params[:, 9:10]
    params_dict['scales'] = params[:, 10:11]
    params_dict['means2D'] = params[:, 11:13]
    return params_dict