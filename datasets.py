import torch
from torch.utils.data import Dataset


class BERTDataset(Dataset):
    def __init__(self, input_ids, mask, labels, label_dict):
        self.input_ids = torch.tensor(input_ids)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.labels = labels
        self.label_dict = label_dict

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.input_ids[idx], self.mask[idx], torch.tensor(self.label_dict[self.labels[idx]])
