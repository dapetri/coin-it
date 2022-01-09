import pandas as pd
from ta.volatility import BollingerBands


def add_indicators(df: pd.DataFrame, indicators: list) -> pd.DataFrame:
    if 'BollingerBands' in indicators:
        bb = BollingerBands(
            close=df['close'],
            window=20,
            window_dev=2
        )
        df['bb_m'] = bb.bollinger_mavg()
        df['bb_h'] = bb.bollinger_hband()
        df['bb_l'] = bb.bollinger_lband()

    return df
