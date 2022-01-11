import pandas as pd
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator


def add_indicator(df: pd.DataFrame, indicators: list) -> pd.DataFrame:
    if 'BollingerBands' in indicators:
        bb = BollingerBands(
            close=df['close'],
            window=20,
            window_dev=2,
            fillna=False
        )
        df['bb_m'] = bb.bollinger_mavg()
        df['bb_h'] = bb.bollinger_hband()
        df['bb_l'] = bb.bollinger_lband()

    if 'MACD' in indicators:
        macd = MACD(
            close=df['close'],
            # default 26
            window_slow=26,
            # default 12
            window_fast=12,
            # default 9
            window_sign=9,
            fillna=False
        )
        df['macd_l'] = macd.macd()
        df['macd_h'] = macd.macd_diff()
        df['macd_s'] = macd.macd_signal()

    if 'RSIIndicator' in indicators:
        rsi = RSIIndicator(
            close=df['close'],
            # default 14
            window=14,
            fillna=False
        )
        df['rsi'] = rsi.rsi()

    return df
