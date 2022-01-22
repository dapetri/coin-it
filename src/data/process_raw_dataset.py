from glob import glob
import os
import sys
from Historic_Crypto import Cryptocurrencies
from numpy import empty
import pandas as pd


default_granularity = 60
default_crypto_currencies = [
    'BTC', 'ETH',
]
default_indicators = [
    'BollingerBands'
]
default_file_directory = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..',
    '..',
    'data',
)


class RawDatasetProcessor:
    def __init__(
        self,
        crypto_currencies: list = default_crypto_currencies,
        granularity: int = default_granularity,
        file_directory: str = default_file_directory,
        indicators: list = default_indicators,
    ) -> None:
        self.crypto_currencies = crypto_currencies
        self.granularity = granularity
        self.file_directory = file_directory
        self.indicators = indicators
        self.base_currency = 'EUR'

    def process_raw_datasets(self):
        pairs = []
        seen = []
        crypto_currencies = self.crypto_currencies + [self.base_currency]
        for coin in crypto_currencies:
            seen.append(coin)
            ps = Cryptocurrencies(
                coin_search=coin,
                verbose=False
            ).find_crypto_pairs()['id']
            for pair in ps:
                cryp = pair.replace(coin, '').replace('-', '')
                if cryp not in seen and cryp in crypto_currencies:
                    pairs.append(pair)
        files = []
        for dir_path, _, file_names in os.walk(self.file_directory + '/raw/'):
            for file_name in file_names:
                if file_name.split('--')[0] in pairs:
                    pairs.remove(file_name.split('--')[0])
                    files.append(pd.read_csv(dir_path + file_name))
        assert(len(pairs) == 0)


def main():
    dataset_processor = RawDatasetProcessor()
    dataset_processor.process_raw_datasets()
    return 0


if __name__ == '__main__':
    sys.exit(main())
