import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset



class LOBDataset(Dataset):
    def __init__(self, data_path, n_steps: int, labels: list):
        data = pd.read_parquet(data_path, engine='pyarrow')
        data.set_index(['seq_ix', 'step_in_seq', 'need_prediction'], inplace=True)

        self.labels = labels
        self.n_seq = data.index.get_level_values(0).max() +1
         #TODO: check if n_steps is given that all sequences have n_steps
        self.n_steps = n_steps
        self.n_features = data.values.shape[1]

        seq_ids = torch.LongTensor(data.index.get_level_values(0).to_numpy())
        step_ids = torch.LongTensor(data.index.get_level_values(1).to_numpy())
        
        features_normalized = data.groupby(level=0).apply(lambda x: (x - x.mean()) / x.std())
        features_normalized.reset_index(level=0, drop=True, inplace=True)

        self.dataset = self.__stack__(seq_ids, step_ids, features_normalized.values)

    def __stack__(self, seq_ids, step_ids, features):

        data = torch.zeros(self.n_seq, self.n_steps, self.n_features)
        data[seq_ids, step_ids] = torch.FloatTensor(features)

        return data


    def __n_sequences__(self):
        return self.dataset.shape[0]
    
    def __len__(self):
        return self.n_steps
    
   
    def __getitem__(self, index):
        
        return index, self.dataset[index]
