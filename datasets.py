import torch
from torch.utils.data import Dataset

label_dict = {
    'a': {'OFF': 0, 'NOT': 1},
    'b': {'TIN': 0, 'UNT': 1, 'NULL': 2},
    'c': {'IND': 0, 'GRP': 1, 'OTH': 2, 'NULL': 3}
}

class HuggingfaceDataset(Dataset):
    def __init__(self, input_ids, mask, labels, task):
        self.input_ids = torch.tensor(input_ids)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.labels = labels
        self.task = task

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        this_label_dict = label_dict[self.task]
        input = self.input_ids[idx]
        mask = self.mask[idx]
        label = torch.tensor(this_label_dict[self.labels[idx]])
        return input, mask, label

class HuggingfaceMTDataset(Dataset):
    def __init__(self, input_ids, mask, labels):
        self.input_ids = torch.tensor(input_ids)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.labels = labels

    def __len__(self):
        return self.labels['a'].shape[0]

    def __getitem__(self, idx):
        input = self.input_ids[idx]
        mask = self.mask[idx]
        label_a = torch.tensor(label_dict['a'][self.labels['a'][idx]])
        label_b = torch.tensor(label_dict['b'][self.labels['b'][idx]])
        label_c = torch.tensor(label_dict['c'][self.labels['c'][idx]])
        return input, mask, [label_a, label_b, label_c]
