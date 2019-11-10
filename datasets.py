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

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """
    Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset.labels))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, id_):
        return dataset.labels[id_]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
