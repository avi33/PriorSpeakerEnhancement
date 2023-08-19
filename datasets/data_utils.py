import torch

def create_dataset(args):
    if args.dataset == 'cmuarctic':        
        from datasets.cmudata import CMUDataset as Dataset        
        train_set = Dataset(root=r"/media/avi/8E56B6E056B6C86B/datasets/ARCTIC8k", 
                            mode='train', 
                            segment_length=args.seq_len, 
                            sampling_rate=args.sampling_rate, 
                            transforms=None)
        
        test_set = Dataset(root=r"/media/avi/8E56B6E056B6C86B/datasets/ARCTIC8k", 
                           mode='test', 
                           segment_length=args.seq_len, 
                           sampling_rate=args.sampling_rate, 
                           augment=None)
        
    elif args.dataset == 'vctk':
        train_set = None
        test_set = None
                
    else:
        raise ValueError("wrong dataset {}".format(args.dataset))
    
    return train_set, test_set