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
default_usecols = [
    'time', 'close'
]


class RawDatasetProcessor:
    def __init__(
        self,
        crypto_currencies: list = default_crypto_currencies,
        granularity: int = default_granularity,
        file_directory: str = default_file_directory,
        indicators: list = default_indicators,
        usecols: list = default_usecols,
    ) -> None:
        self.crypto_currencies = crypto_currencies
        self.granularity = granularity
        self.file_directory = file_directory
        self.indicators = indicators
        self.base_currency = 'EUR'
        self.usecols = usecols

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
        dfiles = []
        for dir_path, _, file_names in os.walk(self.file_directory + '/raw/'):
            for file_name in file_names:
                if file_name.split('--')[0] in pairs:
                    pairs.remove(file_name.split('--')[0])
                    dfiles.append(
                        pd.read_csv(
                            dir_path + file_name,
                            index_col=0,
                            usecols=self.usecols,
                        )
                    )
        assert(len(pairs) == 0)

        df = dfiles[0]
        for dfile in dfiles[1:]:
            df = df.merge(
                right=dfile,
                how='inner',
                on='time',
            )
        print(df)
        print(df.index[0])

    def make_time_continuous(
        self,
        df: pd.DataFrame,
    ):
        for i in range(df.shape[0] - 1):
            t1 = df.index[i]
            t2 = df.index[i+1]

        # TODO: implement checker if time dif bigger than gran


def main():
    dataset_processor = RawDatasetProcessor()
    dataset_processor.process_raw_datasets()
    return 0


if __name__ == '__main__':
    sys.exit(main())
