import pandas as pd
from Core.Indicators import Indicators

class Preprocess:

    @staticmethod
    def compute_all_indicators(data):
        new_data = Indicators(data)
        new_data.BBANDS()
        new_data.DEMA()
        new_data.EMA()
        new_data.HT_TRENDLINE()
        new_data.KAMA()
        new_data.MA()
        new_data.MIDPOINT()
        new_data.MIDPRICE()
        new_data.SAR()
        new_data.SAREXT()
        new_data.SMA()
        new_data.T3()
        new_data.TEMA()
        new_data.TRIMA()
        new_data.WMA()
        new_data.ADX()
        new_data.ADXR()
        new_data.APO()
        new_data.AROON()
        new_data.AROONOSC()
        new_data.BOP()
        new_data.CCI()
        new_data.CMO()
        new_data.DX()
        new_data.MACD()
        new_data.MACDEXT()
        new_data.MACDFIX()
        new_data.MFI()
        new_data.MINUS_DI()
        new_data.MINUS_DM()
        new_data.MOM()
        new_data.PLUS_DI()
        new_data.PLUS_DM()
        new_data.PPO()
        new_data.ROC()
        new_data.ROCP()
        new_data.ROCR()
        new_data.RSI()
        new_data.STOCH()
        new_data.STOCHF()
        new_data.STOCHRSI()
        new_data.TRIX()
        new_data.ULTOSC()
        new_data.WILLR()
        new_data.AD()
        new_data.ADOSC()
        new_data.OBV()
        new_data.TRANGE()
        new_data.ATR()
        new_data.NATR()
        new_data.data.fillna(0, inplace=True)
        return new_data.data

    @staticmethod
    def get_from_crypto_compare_hourly_price_data_to_pandas(hourly_price_raw_data: list) -> pd.DataFrame():
        hourly_price_data = pd.DataFrame.from_dict(hourly_price_raw_data)

        # Set the time columns as index and convert it to datetime
        hourly_price_data.set_index("time", inplace=True)
        hourly_price_data.index = pd.to_datetime(hourly_price_data.index, unit='s')
        hourly_price_data['datetimes'] = hourly_price_data.index
        hourly_price_data['datetimes'] = hourly_price_data['datetimes'].dt.strftime(
            '%Y-%m-%d')
        return hourly_price_data

    @staticmethod
    def rename_columns(df):
        df.rename(columns={'open': 'Open'}, inplace=True)
        df.rename(columns={'high': 'High'}, inplace=True)
        df.rename(columns={'low': 'Low'}, inplace=True)
        df.rename(columns={'close': 'Close'}, inplace=True)
        df.rename(columns={'volumeto': 'Volume'}, inplace=True)
        df.drop(columns="conversionType", inplace=True)
        df.drop(columns="conversionSymbol", inplace=True)
        df.drop(columns="datetimes", inplace=True)
        df.drop(columns="volumefrom", inplace=True)
        return df


