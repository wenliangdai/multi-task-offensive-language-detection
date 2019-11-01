import torch
from torch.utils.data import Dataset


class HuggingfaceDataset(Dataset):
    def __init__(self, input_ids, mask, labels, task):
        self.input_ids = torch.tensor(input_ids)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.labels = labels
        self.task = task
        self.label_dict = {
            'a': {'OFF': 0, 'NOT': 1},
            'b': {'TIN': 0, 'UNT': 1, 'NULL': 2},
            'c': {'IND': 0, 'GRP': 1, 'OTH': 2, 'NULL': 3}
        }

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        label_dict = self.label_dict[self.task]
        input = self.input_ids[idx]
        mask = self.mask[idx]
        label = torch.tensor(label_dict[self.labels[idx]])
        return input, mask, label
