import cv2
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy.signal import find_peaks
from scipy import stats



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
    hist = torch.histc(tensor, bins=bins)
    hist = hist.detach().cpu().numpy()
    max_val = torch.max(tensor)
    min_val = torch.min(tensor)
    width = (max_val.to(torch.float64) - min_val.to(torch.float64))/bins
    width = width.detach().cpu().numpy()
    #saving the histogram an image
    plt.clf()#clearing the figure
    coordi = range(len(hist))*width + min_val.detach().cpu().numpy()
    plt.bar(coordi, hist, width=width)
    #saving to a png file
    plt.savefig('hist.png')

    if calc_peaks:
        #assuming that g gaussinas and less cant represenr an object therfore zero out all places in the histogram that are less than g
        hist[hist<g] = 0
        # Find the indices where the histogram is zero
        zero_indices = np.where(hist == 0)[0]

        # Find the start indices of runs of n consecutive zeros
        diff = np.diff(zero_indices)
        #split_length = diff[np.where(diff > n)[0]] - 1

        start_index = 0
        peaks = []

        for l in diff:
            if l > n:
                max_index_sub_hist = np.argmax(hist[start_index:start_index+l])
                peaks.append(hist[max_index_sub_hist])
            start_index += l


    return hist



def segment_tensor_using_peaks(tensor, th):
    # Ensure the tensor is 1D
    tensor = tensor.flatten()

    # Convert tensor to numpy for histogram
    #tensor_np = tensor.numpy()

    # Calculate the histogram
    #hist, bins = np.histogram(tensor_np, bins=256, range=(0,256))
    hist = create_hist_from_tensor(tensor, 20)

    # Find the peaks of the histogram
    peaks, _ = find_peaks(hist, height=th)

    # Segment the tensor at the peak intensity values
    segmented_tensor = torch.zeros_like(tensor, dtype=torch.bool)
    for peak in peaks:
        segmented_tensor |= (tensor == peak)

    return segmented_tensor



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
    convert_images_to_video('outputDynamicStaticSplitting/spliting50precentOnlyMeans3D', 'test')