import glob
import os
import torch
import shutil

from torch.utils import data
from PIL import Image
import torchvision
import numpy as np
import torch.nn as nn
import torchvision.transforms.functional as TF
import random
import pickle

from models import transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Joint_Video_Dataset(data.Dataset):

    'Characterizes a dataset for PyTorch'
    #input will be a list of videos
    def __init__(self, synthetic_path, real_path, phrase_length = 70, video_length = 300 ):

        self.alphabet = {
            'a' : 2, 'b' : 3, 'c' : 4, 'd' : 5,'e' : 6, 'f' : 7,
            'g' : 8, 'h' : 9, 'i' : 10, 'j' : 11, 'k' : 12, 'l' : 13,
            'm' : 14,'n' : 15,'o' : 16, 'p' : 17, 'q' : 18, 'r' : 19,
            's' : 20,'t' : 21,'u' : 22,'v' : 23,'w' : 24, 'x' : 25,
            'y' : 26,'z' : 27, ' ' : 28, '<EOS>' : 29, '<SOS>' : 0, '<PAD>':1
        }
        self.synthetic_path = synthetic_path
        self.real_path = real_path
        self.synthetic_data_points = []
        self.real_data_points = []

        self.phrase_length = phrase_length #all phrases will be padded this length
        self.video_length = video_length #all videos will be this length.

        #Load synthetic videos
        for phrase in os.listdir(self.synthetic_path):
            syn_fp = os.path.join(self.synthetic_path, phrase)
            syn_npy = os.path.join(syn_fp, 'data.npy')
            self.synthetic_data_points.append( [syn_npy, phrase])

            # if len(self.synthetic_data_points) > 5000:
            #     break

        #Load real videos
        for phrase in os.listdir(self.real_path):
            real_fp = os.path.join(self.real_path, phrase)
            real_npy = os.path.join(real_fp, 'data.npy')
            phrase = ' '.join(phrase.split('_'))
            self.real_data_points.append( [real_npy, phrase])


    def __len__(self):

        length = len(self.synthetic_data_points)
        return length

    def process_videos(self, vid_npy):

        vid_data = np.load(vid_npy)
        vid_tensor = torch.tensor(vid_data)
        vid_len, feat_dim = vid_tensor.shape

        if vid_tensor.shape[0] > self.video_length:
            #sample between
            #range of rand idxs
            frame_idxs = [x for x in range(vid_len)]
            idxs = []
            idxs = sorted(random.sample(frame_idxs, self.video_length))
            return_tensor = vid_tensor[idxs]
        else:
            return_tensor = torch.zeros((self.video_length, feat_dim))
            return_tensor[:vid_len] = vid_tensor

        return return_tensor

    def __getitem__(self, index):
        '''
        Generates one sample of data
        One sample of data is every video's frames stacked sequentially
        X = stacked seq
        y = label
        '''

        syn_npy, syn_phrase = self.synthetic_data_points[index]
        real_index = random.randint(0, len(self.real_data_points)-1)
        real_npy, real_phrase = self.real_data_points[real_index]

        sos = [self.alphabet['<SOS>']]
        eos = [self.alphabet['<EOS>']]
        pad_index = self.alphabet['<PAD>']

        syn_sent = [self.alphabet[x] for x in syn_phrase]
        real_sent = [self.alphabet[x] for x in real_phrase]

        syn_atoi = sos + syn_sent + eos
        real_atoi = sos + real_sent + eos

        syn_video = self.process_videos(syn_npy)
        real_video = self.process_videos(real_npy)

        syn_atoi = syn_atoi + [self.alphabet['<PAD>']]*(self.phrase_length-len(syn_atoi))
        real_atoi = real_atoi + [self.alphabet['<PAD>']]*(self.phrase_length-len(real_atoi))
        syn_atoi = torch.tensor(syn_atoi, dtype = torch.long).view(-1)
        real_atoi = torch.tensor(real_atoi, dtype = torch.long).view(-1)

        data_dict = {
            'synthetic': [syn_video, syn_atoi, syn_phrase],
            'real': [real_video, real_atoi, real_phrase]
            }
        return data_dict


class Raw_Video_Dataset(data.Dataset):

    'Characterizes a dataset for PyTorch'
    #input will be a list of videos
    def __init__(self, data_path, phrase_length = 70, video_length = 300,
            synthetic = True, training = True, frame_drop_percentage = 1., random_seed =  None ):

        # if random_seed:
        #     random.seed(0)
        #     np.random.seed(0)
        #     torch.manual_seed(0)

        self.alphabet = {
            'a' : 2, 'b' : 3, 'c' : 4, 'd' : 5,'e' : 6, 'f' : 7,
            'g' : 8, 'h' : 9, 'i' : 10, 'j' : 11, 'k' : 12, 'l' : 13,
            'm' : 14,'n' : 15,'o' : 16, 'p' : 17, 'q' : 18, 'r' : 19,
            's' : 20,'t' : 21,'u' : 22,'v' : 23,'w' : 24, 'x' : 25,
            'y' : 26,'z' : 27, ' ' : 28, '<EOS>' : 29, '<SOS>' : 0, '<PAD>':1
        }
        self.data_path = data_path
        self.synthetic = synthetic
        self.phrase_length = phrase_length #all phrases will be padded this length
        self.video_length = video_length #all videos will be this length.
        self.training = training
        self.data_points = []
        self.frame_drop_percentage = frame_drop_percentage #we will keep [frame_drop_percentage, 1.0] of the frames. chop some frames for data augmentation

        if self.synthetic:
            self.frame_num = self.frame_num_syn
        else:
            self.frame_num = self.frame_num_real

        print ("Loading in the videos from {}".format(self.data_path))

        #Load synthetic videos
        for idx, phrase in enumerate(os.listdir(self.data_path)):

            vid_fp = os.path.join(self.data_path, phrase)

            if self.synthetic:
                #We will read in the npy file later
                sorted_frames = os.path.join(vid_fp, 'data.npy')
            else:
                instance_ims = [os.path.join(vid_fp, x) for x in os.listdir(vid_fp) if '.png' in x ]
                sorted_frames = sorted(instance_ims, key = self.frame_num)

            if not self.synthetic:
                phrase = ' '.join(phrase.split('_')).lower()

            data_point = [sorted_frames, phrase, vid_fp]
            self.data_points.append(data_point)


        # if not self.synthetic and self.training:
        #     random.shuffle(self.data_points)
        #     self.data_points = self.data_points[:100]
        #
        print (len(self.data_points))
        

    def __len__(self):

        length = len(self.data_points)
        return length

    def frame_num_real(self, x):

        num = x.split('/')[-1].split('.')[0]
        num = num[5:] #get rid of "frame"
        num = int(num)
        return_num = '{:05d}'.format(num)

        return return_num

    def frame_num_syn(self, x):

        num = x.split('/')[-1].split('.')[0]
        num = int(num)
        return '{:04d}'.format(num)

    def transform_one_image(self, image, random_angle, random_hue,  random_contrast, random_brightness, idx, to_flip):

        image = Image.open(image)
        #for transforms
        if self.training:

            image = torchvision.transforms.functional.adjust_contrast(image, random_contrast)
            image = torchvision.transforms.functional.adjust_hue(image, hue_factor = random_hue)
            image = torchvision.transforms.functional.adjust_brightness(image, random_brightness)
            image = torchvision.transforms.functional.rotate(image, angle = random_angle)
            # if to_flip:
            #     image = torchvision.transforms.functional.vflip(image)

            # image.save("{}_sample.png".format(idx))

        if self.training:
            preprocess = torchvision.transforms.Compose([
               torchvision.transforms.ToTensor(),
               torchvision.transforms.Normalize([0.5,0.5, 0.5], [0.5,0.5,0.5]),
               torchvision.transforms.RandomErasing()
            ])
        else:
            preprocess = torchvision.transforms.Compose([
               torchvision.transforms.ToTensor(),
               torchvision.transforms.Normalize([0.5,0.5, 0.5], [0.5,0.5,0.5]),

            ])

        return preprocess(image)

    def process_raw_videos(self, video_frames):


        random_angle = 0
        random_hue = 0
        random_contrast = 1
        random_brightness = 1
        to_flip = False

        if self.training:

            '''
            if random.random() > 0:
                 #perform transformations
                #random_angle = random.uniform(-5, 5)
                random_hue = random.uniform(-0.4, 0.4)
                random_contrast = random.uniform(0.75, 1.25)
                random_brightness = random.uniform(0.5, 1.5)
            '''
            #if random.random() > 0.5:
            #    to_flip = True
            percentage_keep = random.uniform(self.frame_drop_percentage, 1.0)
            aug_length = int (len(video_frames) * percentage_keep)

            frame_idxs = np.arange(0, len(video_frames)).tolist()
            aug_idxs = []
            aug_idxs = sorted(random.sample(frame_idxs, aug_length))
            video_frames = [video_frames[i] for i in aug_idxs]


        frames_transform = [self.transform_one_image(x, random_angle, random_hue,  random_contrast, random_brightness, idx, to_flip) for idx, x in enumerate(video_frames)]
        vid_tensor = torch.stack(frames_transform)

        vid_len = vid_tensor.shape[0]
        if vid_len > self.video_length:
            #sample between
            #range of rand idxs
            frame_idxs = [x for x in range(vid_len)]
            idxs = []
            idxs = sorted(random.sample(frame_idxs, self.video_length))
            return_tensor = vid_tensor[idxs]
        else:
            return_tensor = torch.zeros((self.video_length, vid_tensor.shape[1], vid_tensor.shape[2], vid_tensor.shape[3] ))
            return_tensor[:vid_len] = vid_tensor

        return return_tensor

    def process_synthetic_videos(self, video_frames):

        video_frames = np.load(video_frames)

        if self.training:
            percentage_keep = random.uniform(self.frame_drop_percentage, 1.0)
            aug_length = int (len(video_frames) * percentage_keep)
            frame_idxs = np.arange(0, len(video_frames)).tolist()

            aug_idxs = []
            aug_idxs = sorted(random.sample(frame_idxs, aug_length))
            video_frames = video_frames[aug_idxs]

        vid_tensor = torch.from_numpy(video_frames)

        vid_len = vid_tensor.shape[0]
        if vid_len > self.video_length:
            #sample between
            #range of rand idxs
            frame_idxs = [x for x in range(vid_len)]
            idxs = []
            idxs = sorted(random.sample(frame_idxs, self.video_length))
            return_tensor = vid_tensor[idxs]
        else:
            return_tensor = torch.zeros((self.video_length, vid_tensor.shape[1]))
            return_tensor[:vid_len] = vid_tensor

        return return_tensor

    def __getitem__(self, index):
        '''
        Generates one sample of data
        One sample of data is every video's frames stacked sequentially
        X = stacked seq
        y = label
        '''
        video_f, phrase, fp = self.data_points[index]

        sos = [self.alphabet['<SOS>']]
        eos = [self.alphabet['<EOS>']]
        pad_index = self.alphabet['<PAD>']

        sent = [self.alphabet[x] for x in phrase]

        atoi = sos + sent + eos
        atoi = atoi + [self.alphabet['<PAD>']]*(self.phrase_length-len(atoi))
        atoi_tensor = torch.tensor(atoi, dtype = torch.long).view(-1)

        if self.synthetic:
            video_processed = self.process_synthetic_videos(video_f)
            # inv_idx = torch.arange(video_processed.size(0)-1,-1,  -1).long()
        else:
            video_processed = self.process_raw_videos(video_f)
            # inv_idx = torch.arange(video_processed.size(0)-1, -1, -1).long()


        # or equivalently torch.range(tensor.size(0)-1, 0, -1).long()
        # video_processed = video_processed.index_select(0, inv_idx)
        # or equivalently



        return video_processed, atoi_tensor, fp



if __name__ == '__main__':
    import argparse
    '''
    Testing the dataloaders
    '''

    parser = argparse.ArgumentParser()

    #model hyperparameters
    parser.add_argument("--synthetic_train_data_path", type = str, default = '/data/synthetic_preprocessed_split/train')
    parser.add_argument("--synthetic_test_data_path", type = str, default = '/data/synthetic_preprocessed_split/val')
    parser.add_argument("--real_train_data_path", type = str, default = 'data/data/train/')
    parser.add_argument("--real_test_data_path", type = str, default = 'data/data/test/')

    args = parser.parse_args()

    for vid in os.listdir(args.real_test_data_path):
        vid_length = len(os.listdir(os.path.join(args.real_test_data_path, vid)))

    syn_train_dataset = Raw_Video_Dataset(args.synthetic_train_data_path, synthetic = True, training = True)
    syn_train_dataloader = data.DataLoader(syn_train_dataset, batch_size = 1, num_workers = 0, shuffle = False)

    real_train_dataset = Raw_Video_Dataset(args.real_train_data_path, synthetic = False, training = True)
    real_train_dataloader = data.DataLoader(real_train_dataset, batch_size = 2, num_workers = 0, shuffle = False)
    for x in real_train_dataloader:
        print (x[1].shape)
        exit()
    exit()
