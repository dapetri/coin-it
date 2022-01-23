from datetime import datetime, timedelta
from glob import glob
import os
import sys
from Historic_Crypto import Cryptocurrencies
from numpy import empty
import pandas as pd


default_accepted_error = 60
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
        accepted_error: int = default_accepted_error,
        file_directory: str = default_file_directory,
        indicators: list = default_indicators,
        usecols: list = default_usecols,
    ) -> None:
        self.crypto_currencies = crypto_currencies
        self.accepted_error = accepted_error
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
        # self._error_count(
        #     df=df,
        #     accepted_error=[60, 120, 180, 240, 300]
        # )

        # self._biggest_error(df)

        file_name = '-'.join(crypto_currencies) + '--g' + \
            str(self.accepted_error) + '.csv'
        file_name = self.file_directory + '/processed/' + file_name
        df.to_csv(file_name)

    def make_time_continuous(
        self,
        df: pd.DataFrame,
    ):
        last = 0
        for i in range(df.shape[0] - 1):
            d_earlier = self.transform_date(df.index[i])
            d_later = self.transform_date(df.index[i+1])
            if not self.date_continuous(
                date_earlier=d_earlier,
                date_later=d_later,
                accepted_error=self.accepted_error,
            ):
                last = i
        print(df.index[range(last)])
        return df.drop(
            df.index[range(last)],
        )

        # TODO: implement checker if time dif bigger than gran

    def _biggest_error(
        self,
        df: pd.DataFrame,
    ):
        error_diff = 0
        for i in range(df.shape[0] - 1):
            d_earlier = self.transform_date(df.index[i])
            d_later = self.transform_date(df.index[i+1])
            diff = self.time_difference(d_earlier, d_later)
            if diff > error_diff:
                error_diff = diff
        print(error_diff)

    def _error_count(
        self,
        df: pd.DataFrame,
        accepted_error: list,
    ) -> None:
        print('ERRORS:')
        for error in accepted_error:
            count = 0
            last = ''
            for i in range(df.shape[0] - 1):
                d_earlier = self.transform_date(df.index[i])
                d_later = self.transform_date(df.index[i+1])
                if not self.date_continuous(
                    date_earlier=d_earlier,
                    date_later=d_later,
                    accepted_error=error,
                ):
                    count += 1
                    last = d_later
            print(str(error) + 'sec: ' + str(count) + ', last at: ' + last)

    def transform_date(
        self,
        date: str,
    ) -> str:
        return date.replace(' ', '-').replace(':', '-')[:-3]

    def time_difference(
        self,
        date_earlier: str,
        date_later: str,
    ) -> int:
        d_earlier = [int(i) for i in date_earlier.split(sep='-')]
        d_later = [int(i) for i in date_later.split(sep='-')]
        dt_earlier = datetime(
            year=d_earlier[0],
            month=d_earlier[1],
            day=d_earlier[2],
            hour=d_earlier[3],
            minute=d_earlier[4]
        )
        dt_later = datetime(
            year=d_later[0],
            month=d_later[1],
            day=d_later[2],
            hour=d_later[3],
            minute=d_later[4]
        )
        return (dt_later - dt_earlier).total_seconds()

    def date_continuous(
        self,
        date_earlier: str,
        date_later: str,
        accepted_error: int,
    ) -> bool:
        d_earlier = [int(i) for i in date_earlier.split(sep='-')]
        d_later = [int(i) for i in date_later.split(sep='-')]
        dt_earlier = datetime(
            year=d_earlier[0],
            month=d_earlier[1],
            day=d_earlier[2],
            hour=d_earlier[3],
            minute=d_earlier[4]
        )
        dt_later = datetime(
            year=d_later[0],
            month=d_later[1],
            day=d_later[2],
            hour=d_later[3],
            minute=d_later[4]
        )

        return dt_earlier + timedelta(seconds=accepted_error) >= dt_later


def main():
    dataset_processor = RawDatasetProcessor()
    dataset_processor.process_raw_datasets()
    return 0


if __name__ == '__main__':
    sys.exit(main())
