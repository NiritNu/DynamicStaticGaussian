import cv2
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.signal import find_peaks
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D




## ,aking video from images in folder
def convert_images_to_video(image_folder, video_name):
    n = 27  # number of videos
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    video = []
    for i in range(n):
        video.append(cv2.VideoWriter(video_name + str(i) + '.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)))

    #sort images to different movies by cam_id number , file name is  of shape timestep_*_seq_*_cam_id_*.png
    images.sort(key=lambda f: int(''.join(filter(str.isdigit, f.split('_')[6]))))
    
    # split to m different lists by order
    m = int(len(images)/n)            
    images = [images[i * m:(i + 1) * m] for i in range((len(images) + m - 1) // m)]


    #sort images in each list by timestep number
    for i in range(len(images)):
        images[i].sort(key=lambda f: int(''.join(filter(str.isdigit, f.split('_')[1]))))

    for i, image in enumerate(images):
        for im in image:
            video[i].write(cv2.imread(os.path.join(image_folder, im)))

    cv2.destroyAllWindows()
    for i in range(len(video)):
        video[i].release()

#the function conveet images to video but the images are sorted by timestep and not by cam_id_and an object
def convert_images_to_video2(image_main_folder, video_name):
    n = 27  # number of cameras
    # going throgh all folders in image_main_folder named timestep_* only
    first_time = True
    # going throgh all folders in image_main_folder
    timesteps = os.listdir(image_main_folder)
    timesteps.sort(key=lambda f: int(''.join(filter(str.isdigit, f.split('_')[1]))))
    for folder in timesteps: # gioing throgh all timesteps
        image_folder = os.path.join(image_main_folder, folder)
        #check if folder is named timestep_*
        if image_folder.split('/')[-1].split('_')[0] == 'timestep':
            image_folder = os.path.join(image_folder, 'movment')
            
            images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
            #sort images to different movies by cam_id number , file name is  of shape timestep_1_seq_basketball_cam_id_0_obj_0.png
            images.sort(key=lambda f: int(''.join(filter(str.isdigit, f.split('_')[6]))))
            # find how many objects are in the folder
            #go throg all images and find the number of objects
            num_of_objects = []
            for img in images:
                num_of_objects.append(int(img.split('_')[-1].split('.')[0]))
            objects_ids = np.array(list(set(num_of_objects)))
            object_num = len(objects_ids)
            
            if first_time:
                all_videos = [[[] for _ in range(object_num)] for _ in range(n)]
                first_time = False
            index = 0
            
            for cam_id in range(n):
                all_images_per_object = images[index:index+object_num]
                all_images_per_object.sort(key=lambda f: int(''.join(filter(str.isdigit, f.split('_')[-1].split('.')[0]))))
                for o in range(object_num):
                    all_videos[cam_id][o].append(all_images_per_object[o])
                   
                index += object_num
    
    # write each cam_id and object to a different video
    for cam_id in range(n):
        for o in range(object_num):
            time_step = '_'.join(all_videos[cam_id][o][0].split('_')[0:2])
            frame = cv2.imread(os.path.join(image_main_folder,time_step,'movment',all_videos[cam_id][o][0]))
            height, width, _ = frame.shape
            video = cv2.VideoWriter(video_name + '_cam_id_' + str(cam_id) + '_obj_' + str(o) + '.mp4', cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
            for im in all_videos[cam_id][o]:
                time_step = '_'.join(im.split('_')[0:2])
                video.write(cv2.imread(os.path.join(image_main_folder,time_step,'movment', im)))
            video.release()
    


# a function that upload all png files in folder and sort it in a list by str can_id in the file name
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            images.append(filename)
    images.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return images

## Histograms

# Create a torch vector with two normal distributions
def create_vector(mean1, std1, mean2, std2, size):
    half = size // 2
    # Create first half of the vector with a normal distribution centered around mean1
    vec1 = torch.normal(mean=mean1, std=std1, size=(half,))
    # Create second half of the vector with a normal distribution centered around mean2
    vec2 = torch.normal(mean=mean2, std=std2, size=(size-half,))
    # Concatenate the two halves to create the full vector
    vec = torch.cat((vec1, vec2))
    return vec

#vec = create_vector(3, 1, -3, 1, 1000)

#The Z-score is a measure of how many standard deviations an element is from the mean. Elements with a Z-score greater than a threshold (for example, 3) are considered outliers.
def remove_outliers(data, z_threshold=3):
    z_scores = np.abs(stats.zscore(data))
    data[z_scores > z_threshold] = 0
    return data

#creating a histogram from torch tensor and casting it to numpy array
def create_hist_from_tensor_and_calc_peaks_values(tensor, bins, calc_peaks=False, g=10, n=2):
    max_val = tensor.max().item()
    min_val = tensor.min().item()
    hist = torch.histc(tensor, bins=bins, min=min_val, max=max_val)
    hist = hist.detach().cpu().numpy()
    #width = (max_val.to(torch.float64) - min_val.to(torch.float64))/bins
    #width = width.detach().cpu().numpy()
    width=(max_val-min_val)/bins
    #saving the histogram an image
    plt.clf()#clearing the figure
    #copilot I did not checj:
    bin_edges = np.linspace(min_val, max_val, bins+1)

    # Calculate the x-labels (centers of the bins)
    x_labels = (bin_edges[:-1] + bin_edges[1:]) / 2
    #coordi = range(len(hist))*width + min_val.detach().cpu().numpy()
    plt.bar(x_labels, hist, width,align='center')
    #saving to a png file
    plt.savefig('hist.png')

    if calc_peaks:
        #assuming that g gaussinas and less cant represenr an object therfore zero out all places in the histogram that are less than g
        hist[hist<g] = 0
        # Find the indices where the histogram is zero
        zero_indices = np.where(hist == 0)[0]
        zero_indices = np.insert(zero_indices, 0, 0)

        # Find the start indices of runs of n consecutive zeros
        diff = np.diff(zero_indices)
        #split_length = diff[np.where(diff > n)[0]] - 1

        start_index = 1 #to do : I am not using 0 because it is zero movment and I am not intersted in it
        peaks = []

        #todo: check if i dont need to substract 1 from max_index_sub_hist
        for l in diff:
            if l > n:
                max_index_sub_hist = np.argmax(hist[start_index:start_index+l-1]) + start_index
                peaks.append(x_labels[max_index_sub_hist])
            start_index += l


    return hist, peaks


def segmenting_by_movement(curr_movment,params, bool_index_movment, g=10, n=2):
    
    # Calculate the histogram and peaks
    _, peaks = create_hist_from_tensor_and_calc_peaks_values(curr_movment, int(len(curr_movment)/1000), calc_peaks=True, g=g, n=n)

    curr_movment = curr_movment.detach()
    #expanding curr_movement len(bins) times
    curr_movment_mat= curr_movment.repeat(len(peaks), 1)
    peaks_tensor = torch.tensor(peaks).unsqueeze(1).repeat(1,curr_movment.shape[0]).to(device=curr_movment.device)
    diff_mat = torch.abs(curr_movment_mat - peaks_tensor)
    #create boolean tensor that contains True to the largest value in each coloumn on diff_mat
    bool_tensor = torch.zeros(diff_mat.shape, dtype=torch.bool)
    max_tensor = torch.max(diff_mat, dim=0)
    bool_tensor[max_tensor[1], torch.arange(len(max_tensor[1]))] = True

    obj_params = []
    for obj in range(len(peaks)):
        obj_params.append(params.copy())
        for key in obj_params[-1]:
                if (key != 'cam_m') & (key != 'cam_c'):
                    bool_tensor[obj,:] = bool_tensor[obj,:] & bool_index_movment.to(device=bool_tensor.device)
                    obj_params[-1][key] = obj_params[-1][key][bool_tensor[obj,:]]

    #I have the object divided by speed now I need to divide it by location (obj_params['means3D'])
    
    return obj_params

#I have the object divided by speed now I need to divide it by location (obj_params['means3D'])
'''def segmenting_by_location(obj_params, bins):
    #going throgh all objects with the same speed and dividing them by location
    for obj in range(len(obj_params)):
        location = obj_params[obj]['means3D']
        # Calculate the histogram and peaks 3D location
        # Defune the number of bins for each dimension
        bins = [bins, bins, bins]
        # Compute the 3D histogram
        hist_3d, edges = np.histogramdd(location.detach().cpu().numpy(), bins=bins)
        # Create a 3D plot
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xedges, yedges, zedges = edges
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = dy = 0.5 * np.ones_like(zpos)
    dz = hist_3d.ravel()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    plt.savefig('location_hist.png')'''
        

    
#sorting each element in torch tensor to the closest number of a second input and returning a new tensor that contains the indexes of the closest numbers
def sort_to_bins(tensor, bins):
    new_tensor = torch.zeros(tensor.shape)
    for i in range(len(tensor)):
        min_d   = 1000000
        min_idx = 0
        for j in range(len(bins)):
            if abs(tensor[i]-bins[j]) < min_d:
                min_d = abs(tensor[i]-bins[j])
                min_idx = j
        new_tensor[i] = min_idx 
    return new_tensor



#return a boolean tensor that contains true in the indexes where the original input tensor is close engouh to the input value
def find_close_values(tensor, value, th):
    bool_tensor = torch.zeros(tensor.shape, dtype=torch.bool)
    diff_tensor = torch.abs(tensor-value)
    bool_tensor[diff_tensor<th] = True
    
    return bool_tensor






if __name__ == '__main__':
    #convert_images_to_video('outputDynamicStaticSplitting/spliting50precentOnlyMeans3D', 'test')
    convert_images_to_video2('outputDynamicStaticSplitting/exp1/basketball', 'test2')