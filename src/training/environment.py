import sys
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
import os
from data.dataset import Dataset


default_file_directory = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..',
    '..',
    'data',
    'processed'
)


class Environment:
    def __init__(
        self,
        validation: int = 0.1,
        file_name: str = 'BTC-ETH-EUR--g60.csv',
        file_directory: str = default_file_directory,
    ) -> None:
        self.validation = validation
        self.file_directory = file_directory
        file_path = os.path.join(
            self.file_directory,
            file_name,
        )
        df = pd.read_csv(
            file_path,
            index_col=0,
        )
        dataset = Dataset(df)
        n_val = df.shape[0] * validation
        n_train = df.shape[0] - n_val
        # TODO: continue with torch dataset
        train_set, val_set = random_split(
            dataset=dataset,
            lengths=[n_train, n_val],
        )

    # def step():

    # def reset():


def main():
    env = Environment()
    return 0


if __name__ == '__main__':
    sys.exit(main())
