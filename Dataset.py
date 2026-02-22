import pandas as pd
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

        #TODO: sequences are independent, adjust normalization per sequence
        mean = data.mean()
        std = data.std()
        df = (data - mean) / std
        self.mean = torch.tensor(mean.to_numpy()).reshape(1, -1)
        self.std = torch.tensor(std.to_numpy()).reshape(1, -1)

        seq_ids = torch.LongTensor(data.index.get_level_values(0).to_numpy())
        step_ids = torch.LongTensor(data.index.get_level_values(1).to_numpy())
        features = torch.FloatTensor(data.values)

        self.dataset = self.__stack__(seq_ids, step_ids, features)

    def __stack__(self, seq_ids, step_ids, features):

        data = torch.zeros(self.n_seq, self.n_steps, self.n_features)
        data[seq_ids, step_ids] = features

        return data


    def __n_sequences__(self):
        return self.n_seq
    
    def __len__(self):
        return self.n_steps
    
   
    def __getitem__(self, index):
        
        return index, self.dataset[index]
