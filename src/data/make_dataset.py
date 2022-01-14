from datetime import date, datetime
from os import error, path, remove, walk
import os
import sys
from Historic_Crypto import Cryptocurrencies
from Historic_Crypto import HistoricalData
import pandas as pd
from glob import glob
from argparse import ArgumentParser
import add_indicators
from pprint import pprint

""" Date comparisson based on date format of Historic_Crypto """

# BTC 2017-01-01
# ETH 2017-06-01
# ETC
# BCH 
# LTC
# USDT
# DOGE
# SHIB
# XLM

default_start_date = '2017-06-01-00-00'
default_granularity = 60
default_crypto_currencies = [
    'BTC', 'ETH'
]
default_indicators = [
    'BollingerBands'
]
default_file_directory = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '..', '..', 'data')


class DatasetMaker:
    def __init__(
            self,
            start_date: str = default_start_date,
            crypto_currencies: list = default_crypto_currencies,
            granularity: int = default_granularity,
            file_directory: str = default_file_directory,
            indicators: list = default_indicators,
    ):
        self.crypto_currencies = crypto_currencies
        self.granularity = granularity
        self.start_date = start_date
        self.file_directory = file_directory
        self.indicators = indicators
        self.base_currency = 'EUR'

    def update_indicators(self):
        for (dirpath, _, filenames) in walk(self.file_directory + '/raw'):
            for filename in filenames:
                df = pd.read_csv(dirpath + filename)
                df = add_indicators(df, self.indicators)
                pd.to_csv(self.file_directory + '/processed' + filename)

    def update_raw_datasets(self):
        seen = []
        crypto_currencies = self.crypto_currencies + [self.base_currency]
        for coin in crypto_currencies:
            seen.append(coin)
            pairs = Cryptocurrencies(
                coin_search=coin,
                verbose=False
            ).find_crypto_pairs()['id']
            for pair in pairs:
                cryp = pair.replace(coin, '').replace('-', '')
                if cryp not in seen and cryp in crypto_currencies:
                    # file name without end date
                    file_name = self.file_directory + '/raw/' +\
                        pair + '--g' + str(self.granularity)
                    complete_file_name = glob(file_name + '*')
                    # if file already exist try to keep start day as early as possible
                    if complete_file_name:
                        old = pd.read_csv(
                            complete_file_name[0],
                            index_col=0
                        )
                        start_date = self.transform_date(old.index[0])
                        end_date = self.transform_date(old.index[-1])
                        # self.start_date < start_date for unchanged date bc historic_crypto returns start_date excluded
                        if self.compare_dates(self.start_date, start_date):
                            new_before = self.get_dataset(
                                pair=pair,
                                start_date=self.start_date,
                                end_date=start_date
                            )
                            old = new_before.append(old[1:])
                        new_after = self.get_dataset(
                            pair=pair, 
                            start_date=end_date
                        )
                        new = old.append(new_after)
                        remove(complete_file_name[0])
                    else:
                        new = self.get_dataset_recursive(
                            file_name_suffix=file_name,
                            pair=pair,
                            start_date=self.start_date
                        )
                    start = self.transform_date(str(new.index[0]))
                    now = self.transform_date(str(new.index[-1]))
                    file_name += '--sd' + start + '--ed' + now + '.csv'
                    new.to_csv(file_name)

    def get_dataset_recursive(
        self,
        file_name_suffix: str,
        pair: str,
        start_date: str,
    ) -> pd.DataFrame:
        current_year = int(start_date[0:4])
        current_rest = start_date[4:]
        file_name_suffix_clean = file_name_suffix
        file_name_suffix += '--BUFFERED.csv'
        if datetime.now().year == current_year:
            return self.get_dataset(
                pair=pair,
                start_date=start_date
            )
        if glob(file_name_suffix):
            old = pd.read_csv(
                file_name_suffix,
                index_col=0
            )
            end_date = self.transform_date(old.index[-1])
            new_after = self.get_dataset_recursive(
                file_name_suffix=file_name_suffix_clean,
                pair=pair,
                start_date=end_date
            )
            return old.append(new_after)
        diff = datetime.now().year - current_year 
        assert(diff > 1)
        for i in range(diff - 1):
            current_start = str(current_year + i) + current_rest
            current_end = str(current_year + i + 1) + current_rest
            new = self.get_dataset(
                pair=pair,
                start_date=current_start,
                end_date=current_end
            )
            if glob(file_name_suffix):
                old = pd.read_csv(
                    file_name_suffix,
                    index_col=0
                )
                new = old.append(new)
                remove(file_name_suffix)
            new.to_csv(file_name_suffix)
        new = self.get_dataset(
            pair=pair,
            start_date=str(datetime.now().year) + current_rest
        )
        if glob(file_name_suffix):
            old = pd.read_csv(
                file_name_suffix,
                index_col=0
            )
            new = old.append(new)
        remove(file_name_suffix)
        return new 

    def get_dataset(
            self,
            pair: str,
            start_date: str,
            end_date: str = ''
    ) -> pd.DataFrame:
        return HistoricalData(
            ticker=pair,
            granularity=self.granularity,
            start_date=start_date,
            end_date=end_date,
            verbose=False
        ).retrieve_data()

    """ transform date from '%Y-%m-%d %H:%M:%S' to '%y-%m-%d-%h-%m' """

    def transform_date(self, date: str) -> str:
        return date.replace(' ', '-').replace(':', '-')[:-3]

    """ Date comparisson based on format '%y-%m-%d-%h-%m', if d1 is earlier than d2"""
    # works only if gran is not set on daily (max)
    def compare_dates(self, date_1: str, date_2: str) -> bool:
        d_1 = [int(i) for i in date_1.split(sep='-')]
        d_2 = [int(i) for i in date_2.split(sep='-')]
        return datetime(
            year=d_1[0],
            month=d_1[1],
            day=d_1[2],
            hour=d_1[3],
            minute=d_1[4]
        ) < datetime(
            year=d_2[0],
            month=d_2[1],
            day=d_2[2],
            hour=d_2[3],
            minute=d_2[4]
        )


def main() -> int:
    parser = ArgumentParser(
        description='Reading arguments for DatasetMaker.'
    )
    parser.add_argument(
        '-sd', '--start_date',
        type=str,
        required=False,
        default=default_start_date
    )
    parser.add_argument(
        '-cc', '--crypto_currencies',
        type=str,
        nargs='+',
        required=False,
        default=default_crypto_currencies
    )
    parser.add_argument(
        '-g', '--granularity',
        type=int,
        required=False,
        default=default_granularity
    )
    args = parser.parse_args()
    start_date = args.start_date
    crypto_currencies = args.crypto_currencies
    granularity = args.granularity
    dataset_maker = DatasetMaker(
        start_date=start_date,
        crypto_currencies=crypto_currencies,
        granularity=granularity
    )
    pprint(vars(dataset_maker))
    dataset_maker.update_raw_datasets()
    # dataset_maker.update_indicators()
    return 0


if __name__ == '__main__':
    sys.exit(main())
