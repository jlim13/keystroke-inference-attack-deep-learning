import glob
import os
import shutil
import random
from distutils.dir_util import copy_tree
from multiprocessing import Pool
from PIL import Image

random.seed(1234)


def frame_num(x):

    num = x.split('/')[-1].split('.')[0]
    num = num[5:] #get rid of "frame"
    num = int(num)
    return_num = '{:05d}'.format(num)

    return return_num

def preprocess(source_directory, target_directory):


    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    source_directory = os.path.join(source_directory)

    instance_ims = [os.path.join(source_directory, x) for x in os.listdir(source_directory) if not '.DS_Store' in x ]
    sorted_frames = sorted(instance_ims, key = frame_num)

    sorted_pil_ims = [Image.open(x) for x in sorted_frames]
    for img in sorted_pil_ims:
       new_img = img.resize((100,200)) #width x height
       box = ( 0,  100,  100, 200) #(left, upper, right, lower)
       cropped_img = new_img.crop(box)

       original_fp = img.filename
       img_name = original_fp.split('/')[-1]
       new_img = os.path.join(target_directory, img_name)
       cropped_img.save(new_img)
#below is to split 10k words, 1 instance each

#list all data files
#randomize
#split by idxs

all_vids = []
data_path =  '../data/raw_real_vids'
train_path = '../data/real_life_videos_preprocessed/train'
val_path = '../data/real_life_videos_preprocessed/test'

#data_path = '/data/synthetic_videos_regression_preprocess'
#train_path = '/data/synthetic_videos_regression_preprocess_split/train'
#val_path = '/data/synthetic_videos_regression_preprocess_split/val'

for word_vid in os.listdir(data_path):
    word_path = os.path.join(data_path, word_vid)
    all_vids.append(word_path)

random.shuffle(all_vids)
val_idx = int(len(all_vids) * .15)
val_list = all_vids[0:val_idx]
train_list = all_vids[val_idx:]

count = 0

for src_type, dest_type in zip([val_list,  train_list], [val_path,train_path]):
    dest_dir = dest_type
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for src_vid in src_type:

        src_dir = '/'.join(src_vid.split('/')[-1:])
        dest_video_path = os.path.join(dest_dir, src_dir)

        #copy_tree(src_vid, dest_video_path)
        preprocess(src_vid, dest_video_path)

        count += 1

        print ('{} out of {}'.format(count, len(all_vids)))
