from datetime import datetime
from os import error, remove
from Historic_Crypto import Cryptocurrencies
from Historic_Crypto import HistoricalData
import pandas as pd
from glob import glob

""" Date comparisson based on date format of Historic_Crypto """


class DatasetMaker:
    def __init__(
            self,
            start_date: str,
            file_directory: str,
            crypto_currencies: list = [
                'BTC', 'ETH', 'SOL', 'ADA', 'SHIB', 'LTC'
            ],
            granularity: int = 60,
    ):
        self.crypto_currencies = crypto_currencies
        self.granularity = granularity
        self.start_date = start_date
        self.file_directory = file_directory

    def update_datasets(self):
        seen = []
        crypto_currencies = self.crypto_currencies + ['EUR']
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
                    file_name = self.file_directory + \
                        pair + '--g' + str(self.granularity)
                    complete_file_name = glob(file_name + '*')
                    if complete_file_name:
                        old = pd.read_csv(
                            complete_file_name[0], index_col=0)
                        start_date = self.transform_date(old.index[0])
                        end_date = self.transform_date(old.index[-1])
                        if self.compare_dates(self.start_date, start_date):
                            new_before = self.get_dataset(
                                pair=pair,
                                start_date=self.start_date,
                                end_date=start_date)
                            new_after = self.get_dataset(
                                pair=pair, start_date=end_date)
                            new = new_before.append(
                                old.iloc[1:-1]).append(new_after)
                        else:
                            if self.compare_dates(self.start_date, end_date):
                                new_after = self.get_dataset(
                                    pair=pair, start_date=end_date)
                                new = old.loc[start_date:].iloc[:-
                                                                1].append(new_after)
                            else:
                                new = self.get_dataset(
                                    pair=pair, start_date=end_date)
                        remove(complete_file_name[0])
                    else:
                        new = self.get_dataset(
                            pair=pair, start_date=self.start_date)
                    now = self.transform_date(str(new.index[-1]))
                    file_name += '--sd' + self.start_date + '--ed' + now + '.csv'
                    new.to_csv(file_name)

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
