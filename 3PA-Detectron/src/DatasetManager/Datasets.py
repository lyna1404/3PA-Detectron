import numpy as np
from torch.utils.data import Dataset

class MaskedDataset(Dataset):
    def __init__(self, features: np.ndarray, true_labels: np.ndarray, mask=True, pseudo_labels: np.ndarray = None):
        self.features = features
        self.true_labels = true_labels
        self.mask = mask
        self.pseudo_labels = pseudo_labels
        self.indices = np.arange(len(self.dataset_x))
        self.original_indices = self.indices.copy()

    def __getitem__(self, index):
        index = self.indices[index]
        x = self.features[index]
        y = self.true_labels[index]
        if self.pseudo_labels is not None:
            y_hat = self.pseudo_labels[index]
        else:
            y_hat = y
        return x, y_hat, y, self.mask

    def refine(self, mask: np.ndarray):
        self.indices = self.indices[mask]

    def original(self):
        return MaskedDataset(self.features, self.true_labels, mask=False, pseudo_labels=self.pseudo_labels)

    def reset_index(self):
        self.indices = self.original_indices.copy()

    def __len__(self):
        return len(self.indices)
