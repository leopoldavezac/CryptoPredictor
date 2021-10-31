


from pandas import DataFrame

from crypto_predictor.feature_engineering import get_window_level_df


def test_get_window_level_df():

    CRYPTOS_DF = DataFrame(
        [
            [1001022, 23, 1.3e7],
            [1001023, 12, 1.2e7],
            [1001024, 45, 1.8e7],
            [1001025, 35, 1.1e7],
            [1001026, 78, 1.6e7],
            [1001027, 89, 1.7e7]
        ],
        columns=['timestamp', 'price', 'volume']
    )

    WINDOW_SIZE = 3

    EXPECTED = DataFrame(
        [
            [23, 1.3e7, 12, 1.2e7, 45, 1.8e7, 35],
            [12, 1.2e7, 45, 1.8e7, 35, 1.1e7, 78],
            [45, 1.8e7, 35, 1.1e7, 78, 1.6e7, 89]
        ],
        columns=[
            'price_at_window_index_0',
            'volume_at_window_index_0',
            'price_at_window_index_1',
            'volume_at_window_index_1',
            'price_at_window_index_2',
            'volume_at_window_index_2',
            'price_at_window_index_3'
            ]
    )

    obtained = get_window_level_df(CRYPTOS_DF, WINDOW_SIZE)

    assert assert_df(EXPECTED, obtained)


def assert_df(expected, obtained):

    if expected.columns.tolist() != obtained.columns.tolist():
        return False
    
    return (expected.values == obtained.values).all()

