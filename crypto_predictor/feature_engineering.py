from typing import List

from glob import iglob

from numpy import ndarray, concatenate, zeros, nan

from pandas import DataFrame, read_csv, concat

WINDOW_SIZE = 10

def main() -> None:

    symbols = get_cryptos_symbol()

    cryptos_features_df = []
    for symbol in symbols:
        raw_df = load_raw(symbol)
        features_df = get_window_level_df(raw_df, WINDOW_SIZE)
        features_df['symbol'] = symbol
        cryptos_features_df.append(features_df)

    cryptos_features_df = consolidate(cryptos_features_df)
    save_features(cryptos_features_df)


def get_cryptos_symbol() -> List[str]:
    
    symbols = []
    for f_nm in iglob('./data/raw/*csv'):
        symbols.append(f_nm[11:-4])

    return symbols

def load_raw(symbol:str) -> DataFrame:

    crypto_df = read_csv(
        './data/raw/%s.csv' % symbol,
        usecols=['unix', 'close', 'Volume USDT']
        )
    crypto_df.columns = ['timestamp', 'price', 'volume']

    crypto_df.sort_values('timestamp', inplace=True)

    return crypto_df


def consolidate(cryptos_df: List[DataFrame]) -> DataFrame:
    
    return concat(cryptos_df, axis=0)   


def get_window_level_df(crypto_df:DataFrame, window:int) -> DataFrame:

    features_df = DataFrame()

    for window_index in range(window+1):

        window_var_nms = get_window_var_nms(window_index)

        features_df[window_var_nms] = get_rolled(
            crypto_df[['price', 'volume']].values,
            window_index
            )

    features_df.drop(columns=window_var_nms[1], inplace=True)
    features_df.dropna(axis=0, inplace=True)

    return features_df

def get_window_var_nms(window_index:int) -> List[str]:

    price_var_nm = 'price_at_window_index_%d' % window_index
    volume_var_nm = 'volume_at_window_index_%d' % window_index
    
    return [price_var_nm, volume_var_nm]

def get_rolled(values: ndarray, offset:int) -> ndarray:

    nan_rows = zeros((offset, 2))
    nan_rows[:] = nan

    rolled_values = values[offset:, :]
    rolled_values = concatenate([rolled_values, nan_rows])

    return rolled_values

def save_features(df:DataFrame) -> None:
    
    df.to_csv('./data/feature_store/feature_store.csv', index=False)


if __name__ == '__main__':
    main()