import pandas as pd
from torch.utils.data import Dataset


class Dataset(Dataset):

    def __init__(
        self,
        data: pd.DataFrame
    ):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        state = self.data[index]

        return state
