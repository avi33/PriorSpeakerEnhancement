import os
import torch
import torchaudio
torchaudio.set_audio_backend('sox_io')
import torch.utils.data
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import random
import glob
import json
import sys
sys.path.append(".")
from datasets.audio_augs import AudioAugs


class CMUDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """
    def __init__(self, data_path, mode, segment_length, sampling_rate, transforms=None, pairs=False, same_segment=False):
        self.data_path = data_path
        self.mode = mode
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        self.audio_files = self.__parse_data()                
        self.speakers = self.get_speakers(self.audio_files)
        self.transform = AudioAugs(fs=sampling_rate, k_augs=transforms) if transforms else None
        self.num_speakers = len(self.speakers)
        self.spk2idx = dict(zip(self.speakers, range(self.num_speakers)))
        self.pairs = pairs
        self.same_segment = same_segment

    def __parse_data(self):
        with open(self.data_path + '/' + self.mode + '.json', 'r') as f:
            data = json.load(f)
            f_names = [data[s] for s in data.keys()]
            data = [ff for f in f_names for ff in f]
        return data

    def __getitem__(self, index):
        filename = self.audio_files[index]
        spk = filename.split('/')[-3].split('_')[-2]
        spk_idx = self.spk2idx[spk]        
        audio, fs = torchaudio.load(filename)
        audio.squeeze_(0)
        audio = audio/(audio.abs().max() + 1e-8)
        audio = self._get_segment(audio)
        
        if self.pairs:
            if self.transform is not None:
                if self.same_segment:
                    audio = self._get_segment(audio)
                    audio_a = self.transform(audio.clone())
                else:
                    audio_a = self.transform(audio.clone())
                    audio = self._get_segment(audio)
                    audio_a = self._get_segment(audio_a)
                return audio, spk, audio_a
        else:
            if self.transform is not None:
                audio = self.transform(audio)
                return audio, spk_idx

    def __len__(self):
        return len(self.audio_files)

    def _get_segment(self, audio):
        # Take segment
        if audio.size(-1) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start : audio_start + self.segment_length]
        else:
            audio = F.pad(
                audio, (0, self.segment_length - audio.size(0)), "constant"
            ).data
        return audio        
    
    @staticmethod
    def get_speakers(fnames):
        spks = [os.path.dirname(f).split('/')[-2].split('_')[-2] for f in fnames]
        spks = list(set(spks))        
        return spks

    @staticmethod
    def split_data(data_path, split=None):
        #split_by_speaker - open set
        #split_by_samples - close set - stratified per speaker
        fnames = glob.glob(data_path + '/**/wav/*.wav')
        fnames = [f for f in fnames if os.path.isfile(f)]
        speakers = list(set([f.split('/')[-3].split('_')[-2] for f in fnames]))        
        train_portion = 0.8
        if split == 'samples':
            pass    
        elif split == 'speaker':
            idx_train = np.random.permutation(len(speakers))
            speakers_train = [speakers[i] for i in idx_train[:int(train_portion*len(speakers))]]
            speakers_test = set(speakers_train) ^ set(speakers)
            train_data = dict.fromkeys(speakers_train, [])
            test_data = dict.fromkeys(speakers_test, [])        
            for f in fnames:
                spk = f.split('/')[-3].split('_')[-2]
                if spk in speakers_train:
                    train_data[spk].append(f)
                elif spk in speakers_test:
                    test_data[spk].append(f)
                else:
                    ValueError
        else:
            ValueError

        with open(data_path + '/train.json', 'w') as f:
            json.dump(train_data, f, indent=4)

        with open(data_path + '/test.json', 'w') as f:
            json.dump(test_data, f, indent=4)
        print("DONE!")
        return True

    
if __name__ == "__main__":
    data_path = r"/media/avi/54561652561635681/datasets/ARCTIC"
    CMUDataset.split_data(data_path, split='speaker')
    # D = CMUDataset(root=r"/media/avi/8E56B6E056B6C86B/datasets/ARCTIC8k", 
    #                        mode='train', 
    #                        segment_length=8000, 
    #                        sampling_rate=8000, 
    #                        augment=None,
    #                        trim=False)
    # DD = torch.utils.data.DataLoader(D, batch_size=64, shuffle=False, drop_last=False, num_workers=0, pin_memory=False)
    # for i, (x, s) in enumerate(D):        
    #     if x.shape[-1] != 8000:
    #         print(i, x.shape)