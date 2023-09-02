import torch

def create_dataset(args):
    if args.dataset == 'cmuarctic':
        data_path = r"/media/avi/54561652561635681/datasets/ARCTIC"
        from datasets.cmudata import CMUDataset as Dataset
        train_set = Dataset(data_path=data_path, 
                            mode='train', 
                            segment_length=args.seq_len, 
                            sampling_rate=args.sampling_rate,
                            pairs=False,
                            same_segment=False,
                            transforms=args.augs_signal + args.augs_noise)
        
        test_set = Dataset(data_path=data_path, 
                           mode='test', 
                           segment_length=args.seq_len, 
                           sampling_rate=args.sampling_rate, 
                           pairs=False,
                           same_segment=False,
                           transforms=None)
        
    elif args.dataset == 'vctk':
        train_set = None
        test_set = None
                
    else:
        raise ValueError("wrong dataset {}".format(args.dataset))
    
    return train_set, test_set