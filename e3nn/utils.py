from collections import defaultdict
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
import os
from typing import Any, Dict, List, Tuple, Union

DSSP_CODES = ['H', 'G', 'I', 'E', 'O', 'S', 'B', 'T', '-', 'L']
AAs = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'B', 'Z']
N_FEATURES = len(AAs) + len(DSSP_CODES)

class ProteinLoader(Dataset):
    """
    ref: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    Loads a list of protein representations and their corresponding ground-truth drmsds.

    Inheritance:
        Dataset:

    Args:
        directory (str):
        representation (str='Euclidean'):

    >>> 
    """
    
    def __init__(self, dir: str, representation: str='Euclidean', max_length: int=-1):
        """
        Args:
            self (undefined):
            directory (str):
            representation (str='Euclidean'):
        """
        self.dir = dir
        self.labels = pd.read_csv(os.path.join(dir, 'labels.csv'))
        self.rep = representation
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i: Union[int, str]):
        if torch.is_tensor(i):
            i = i.tolist()
        
        if isinstance(i, int):
            name = self.labels.iloc[i, 0]
        elif isinstance(i, str):
            name = i
            i = self.labels[self.labels == i].index[0]
        else:
            raise TypeError(f'Type ({type(i)}) not supported by ProteinLoader.')

        dpath = os.path.join(self.dir, self.rep, f'{name}.npy')
        fpath = os.path.join(self.dir, 'Features', f'{int(name[1:name.index("_")])}.csv')
        coords = np.load(dpath)
        if os.path.isfile(fpath):
            with open(fpath, 'r') as f:
                lines = f.readlines()
            features = read_features(lines)  # concatenated 1-hots for each position
        else:
            features = torch.from_numpy(np.ones_like(coords))[:, 0:1]  # a single '1' at each position
        if (self.max_length != -1) and (len(coords) > self.max_length):
            coords = coords[:self.max_length]
        drmsd = self.labels.iloc[i, 1]
        return {'name': name, 'coords': torch.from_numpy(coords), 'drmsd': torch.tensor(drmsd), 'features': features}

    def __contains__(self, key: Any):
        if isinstance(key, str):
            return (self.labels['protein.iteration'] == key).any()
        else:
            raise TypeError(f'Type ({type(i)}) not supported by ProteinLoader.')
        
def _collate(batch: Dict[str, Union[str, Tensor]]) -> List[Union[Tensor, List[str]]]:
    ret = defaultdict(list)
    for sample in batch:
        ret['coords'].append(sample['coords'])
        ret['features'].append(sample['features'])
        ret['length'].append(torch.tensor(sample['coords'].shape[0]))
        ret['drmsd'].append(sample['drmsd'])
        ret['name'].append(sample['name'])
    for k in ['coords', 'features']:
        ret[k] = torch.nn.utils.rnn.pad_sequence(ret[k], batch_first=True)
    for k in ['length', 'drmsd']:
        ret[k] = torch.stack(ret[k])
    return ret

def read_features(lines: List[str]) -> torch.Tensor:
    """ Takes a list of lines read from a file containing features with the columns 'DSSP' and 'Sequence' and outputs
        a concatenated list of one-hot vectors that is consistent no matter what parameters are passed.
    """
    cols = {k: i for (i, k) in enumerate(lines[0].strip().split(','))}
    features = np.zeros((len(lines) - 1, len(AAs) + len(DSSP_CODES)))
    for i, line in enumerate(lines[1:]):
        l = line.strip().split(',')
        # index in AAs of the sequence of this residue
        features[i, AAs.index(l[cols['Sequence']])] = 1.
        features[i, len(AAs) + DSSP_CODES.index(l[cols['DSSP']])] = 1.
    return torch.from_numpy(features)
    

def mask_by_len(inputs: Tensor, lens: Tensor) -> Tensor:
    ml = torch.max(lens)
    masks = torch.arange(ml)[None, :] < lens[:, None]
    return inputs * masks[:, :, None]

def get_data_loader(dir: str, representation: str='Euclidean', batch_size: int=1, max_length: int=100) -> Tuple[Dataset, DataLoader]:
    """
    Description of get_data_loader

    Args:
        dir (str):
        representation (str='Euclidean'):

    Returns:
        DataLoader

    """
    loader = ProteinLoader(dir=dir, representation=representation, max_length=max_length)
    return loader, DataLoader(loader, batch_size=batch_size, shuffle=True, collate_fn=_collate, pin_memory=True)

if __name__ == '__main__':
    d = '/home/wlg/development/HMS/protein_geometry/data/representations/150_1500'
    dataset, data_loader = get_data_loader(d)
    for batch in data_loader:
        pass