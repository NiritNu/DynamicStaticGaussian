import cv2
import os

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

if __name__ == '__main__':
    convert_images_to_video('outputDynamicStaticSplitting/spliting50precentOnlyMeans3D', 'test')