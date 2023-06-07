import talib
import pandas as pd


class Indicators:

    def __init__(self, data):
        self.data = data

    def CCI(self, ndays: int = 20):
        """Commodity Channel Index"""

        cci = talib.CCI(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=ndays)
        cci = pd.Series(cci,
                        name='CCI')
        self.data = self.data.join(cci)
        return self.data

    def ROC(self, n: int = 10):
        """Rate of Change"""
        N = self.data['Close'].diff(n)
        D = self.data['Close'].shift(n)
        ROC = pd.Series(N / D, name='ROC')
        self.data = self.data.join(ROC)
        return self.data

    def ROCP(self, timeperiod: int = 10):
        """Rate of change Percentage: (price-prevPrice)/prevPrice"""
        rocp = pd.Series(talib.ROCP(self.data['Close'], timeperiod), name='ROCP')
        self.data = self.data.join(rocp)
        return self.data

    def ROCR(self, timeperiod: int = 10):
        """Rate of change ratio: (price/prevPrice)"""
        rocr = pd.Series(talib.ROCR(self.data['Close'], timeperiod), name="ROCR")
        self.data = self.data.join(rocr)
        return self.data

    def RSI(self, timeperiod: int = 14):
        """Relative Strength Index"""
        rsi = pd.Series(talib.RSI(self.data['Close'], timeperiod=timeperiod), name='RSI')
        self.data = self.data.join(rsi)
        return self.data

    def MINUS_DI(self, timeperiod: int = 14):
        """Minus Directional Indicator"""
        minus_di = pd.Series(talib.MINUS_DI(self.data['High'], self.data['Low'], self.data['Close'], timeperiod),
                             name='MINUS_DI')
        self.data = self.data.join(minus_di)
        return self.data

    def PLUS_DI(self, timeperiod: int = 14):
        """PLUS Directional Indicator"""
        plus_di = pd.Series(talib.PLUS_DI(self.data['High'], self.data['Low'], self.data['Close'], timeperiod),
                            name='PLUS_DI')
        self.data = self.data.join(plus_di)
        return self.data

    def ADOSC(self, fastperiod: int = 3, slowperiod: int = 10):
        """Chaikin A/D Oscillator"""
        adosc = pd.Series(
            talib.ADOSC(self.data['High'], self.data['Low'], self.data['Close'], self.data['Volume'], fastperiod,
                        slowperiod), name="ADOSC")
        self.data = self.data.join(adosc)
        return self.data

    def AD(self):
        """Accumulation/Distribution"""
        ad = pd.Series(talib.AD(self.data['High'], self.data['Low'], self.data['Close'], self.data['Volume']),
                       name='AD')
        self.data = self.data.join(ad)
        return self.data

    def ADX(self, timeperiod: int = 14):
        """Average Directional Movement Index"""
        adx = pd.Series(talib.ADX(self.data['High'], self.data['Low'], self.data['Close'], timeperiod), name='ADX')
        self.data = self.data.join(adx)
        return self.data

    def ADXR(self, timeperiod: int = 14):
        """Average Directional Movement Index Rating"""
        adxr = pd.Series(talib.ADXR(self.data['High'], self.data['Low'], self.data['Close'], timeperiod), name='ADXR')
        self.data = self.data.join(adxr)
        return self.data

    def APO(self, fastperiod: int = 12, slowperiod: int = 26, matype: int = 0):
        """Absolute Price Oscillator"""
        apo = pd.Series(talib.APO(self.data['Close'], fastperiod, slowperiod, matype), name='APO')
        self.data = self.data.join(apo)
        return self.data

    def BOP(self):
        """Balance Of Power"""
        bop = pd.Series(talib.BOP(self.data['Open'], self.data['High'], self.data['Low'], self.data['Close']),
                        name="BOP")
        self.data = self.data.join(bop)
        return self.data

    def BBANDS(self, window: int = 20):
        """Compute the Bollinger Bands
        recommended : window = 20"""
        MA = self.data.Close.rolling(window).mean()
        SD = self.data.Close.rolling(window).std()
        self.data['UpperBB'] = MA + (2 * SD)
        self.data['LowerBB'] = MA - (2 * SD)
        return self.data

    def SMA(self, ndays: int = 20):
        """Simple Moving Average"""
        SMA = pd.Series(self.data['Close'].rolling(ndays).mean(), name='SMA')
        self.data = self.data.join(SMA)
        return self.data

    def EWMA(self, ndays: int = 20):
        """Exponentially-weighted Moving Average"""
        EMA = pd.Series(self.data['Close'].ewm(span=ndays, min_periods=ndays - 1).mean(),
                        name='EWMA_' + str(ndays))
        self.data = self.data.join(EMA)
        return self.data

    def ATR(self, n: int = 14):
        """Average True Range"""
        atr = pd.Series(talib.ATR(self.data['High'], self.data['Low'], self.data['Close'], n), name="ATR")
        self.data = self.data.join(atr)
        return self.data

    def NATR(self, timeperiod: int = 14):
        """Normalized Average True Range"""
        natr = pd.Series(
            talib.NATR(self.data['High'], self.data['Low'], self.data['Close'], timeperiod) / self.data['Close'] * 100,
            name='NATR')
        self.data = self.data.join(natr)
        return self.data

    def OBV(self):
        """On Balance Volume"""
        obv = pd.Series(talib.OBV(self.data['Close'], self.data['Volume']), name='OBV')
        self.data = self.data.join(obv)
        return self.data

    def AROON(self, lb: int = 25):
        """Aroon Indicator"""
        self.data['Aroon Up'] = 100 * self.data.High.rolling(lb + 1).apply(lambda x: x.argmax()) / lb
        self.data['Aroon Down'] = 100 * self.data.Low.rolling(lb + 1).apply(lambda x: x.argmin()) / lb
        return self.data

    def AROONOSC(self, timeperiod: int = 25):
        """Aroon Oscillator"""
        aroonosc = pd.Series(talib.AROONOSC(self.data['High'], self.data['Low'], timeperiod), name="AROONOSC")
        self.data = self.data.join(aroonosc)
        return self.data

    def MACD(self, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9):
        """Moving Average Convergence/Divergence"""
        macd, macdsignal, macdhist = talib.MACD(self.data['Close'], fastperiod, slowperiod, signalperiod)
        self.data = self.data.join(pd.Series(macd, name='MACD'))
        self.data = self.data.join(pd.Series(macdsignal, name='MACD SIGNAL'))
        self.data = self.data.join(pd.Series(macdhist, name='MACD HIST'))
        return self.data

    def MACDEXT(self, fastperiod: int = 12, fastmatype: int = 0, slowperiod: int = 26, slowmatype: int = 0, signalperiod: int = 9, signalmatype: int = 0):
        """MACD with controllable MA type"""
        macd, macdsignal, macdhist = talib.MACDEXT(self.data['Close'], fastperiod, fastmatype, slowperiod, slowmatype,
                                                   signalperiod, signalmatype)
        self.data = self.data.join(pd.Series(macd, name='MACDEXT'))
        self.data = self.data.join(pd.Series(macdsignal, name='MACDEXT SIGNAL'))
        self.data = self.data.join(pd.Series(macdhist, name='MACDEXT HIST'))
        return self.data

    def MACDFIX(self, signalperiod: int = 9):
        """Moving Average Convergence/Divergence Fix 12/26"""
        macd, macdsignal, macdhist = talib.MACDFIX(self.data['Close'], signalperiod)
        self.data = self.data.join(pd.Series(macd, name='MACDFIX'))
        self.data = self.data.join(pd.Series(macdsignal, name='MACDFIX SIGNAL'))
        self.data = self.data.join(pd.Series(macdhist, name='MACDFIX HIST'))
        return self.data

    def MFI(self, timeperiod: int = 14):
        """Money Flow Index.
        Formula:
        MFI = 100 - (100 / (1 + PMF / NMF))
        """
        mfi = talib.MFI(self.data['High'], self.data['Low'], self.data['Close'], self.data['Volume'], timeperiod)
        self.data = self.data.join(pd.Series(mfi, name='MFI'))
        return self.data

    def MINUS_DM(self, timeperiod: int = 14):
        """Minus Directional Movement"""
        minus_dm = pd.Series(talib.MINUS_DM(self.data['High'], self.data['Low'], timeperiod), name="MINUS_DM")
        self.data = self.data.join(minus_dm)
        return self.data

    def MOM(self, timeperiod: int = 10):
        """Momentum"""
        mom = pd.Series(talib.MOM(self.data['Close'], timeperiod), name='MOM')
        self.data = self.data.join(mom)
        return self.data

    def PLUS_DM(self, timeperiod: int = 14):
        """Plus Directional Movement"""
        plus_dm = pd.Series(talib.PLUS_DM(self.data['High'], self.data['Low'], timeperiod), name="PLUS_DM")
        self.data = self.data.join(plus_dm)
        return self.data

    def PPO(self, fastperiod: int = 12, slowperiod: int = 26, matype: int = 0):
        """Percentage Price Oscillator"""
        ppo = pd.Series(talib.PPO(self.data['Close'], fastperiod, slowperiod, matype), name='PPO')
        self.data = self.data.join(ppo)
        return self.data

    def STOCH(self, fastk_period: int = 5, slowk_period: int = 3, slowk_matype: int = 0, slowd_period: int = 3, slowd_matype: int = 0):
        """Stochastic"""
        slowk, slowd = talib.STOCH(self.data['High'], self.data['Low'], self.data['Close'], fastk_period, slowk_period,
                                   slowk_matype, slowd_period, slowd_matype)
        self.data = self.data.join(pd.Series(slowk, name="STOCH SLOWK"))
        self.data = self.data.join(pd.Series(slowd, name="STOCH SLOWD"))
        return self.data

    def STOCHF(self, fastk_period: int = 5, fastd_period: int = 3, fastd_matype: int = 0):
        """Stochastic Fast"""
        fastk, fastd = talib.STOCHF(self.data['High'], self.data['Low'], self.data['Close'], fastk_period, fastd_period,
                                    fastd_matype)
        self.data = self.data.join(pd.Series(fastk, name="STOCHF FASTK"))
        self.data = self.data.join(pd.Series(fastd, name="STOCHF FASTD"))
        return self.data

    def STOCHRSI(self, timeperiod: int = 14, fastk_period: int = 5, fastd_period: int = 3, fastd_matype: int = 0):
        """Stochastic Relative Strength Index"""
        fastk, fastd = talib.STOCHRSI(self.data['Close'], timeperiod, fastk_period, fastd_period, fastd_matype)
        self.data = self.data.join(pd.Series(fastk, name="STOCHRSI FASTK"))
        self.data = self.data.join(pd.Series(fastd, name="STOCHRSI FASTD"))
        return self.data

    def TRIX(self, timeperiod: int = 30):
        """1-day Rate-Of-Change (ROC) of a Triple Smooth EMA"""
        trix = pd.Series(talib.TRIX(self.data['Close'], timeperiod), name='TRIX')
        self.data = self.data.join(trix)
        return self.data

    def ULTOSC(self, timeperiod1: int = 7, timeperiod2: int = 14, timeperiod3: int = 28):
        """Ultimate Oscillator"""
        ultosc = pd.Series(
            talib.ULTOSC(self.data['High'], self.data['Low'], self.data['Close'], timeperiod1, timeperiod2,
                         timeperiod3), name='ULTOSC')
        self.data = self.data.join(ultosc)
        return self.data

    def WILLR(self, timeperiod: int = 14):
        """Williams' %R"""
        willr = pd.Series(talib.WILLR(self.data['High'], self.data['Low'], self.data['Close'], timeperiod),
                          name='WILLR')
        self.data = self.data.join(willr)
        return self.data

    def CMO(self, timeperiod: int = 14):
        """Chande Momentum Oscillator"""
        cmo = pd.Series(talib.CMO(self.data['Close'], timeperiod), name='CMO')
        self.data = self.data.join(cmo)
        return self.data

    def DX(self, timeperiod: int = 14):
        """Directional Movement Index"""
        dx = pd.Series(talib.DX(self.data['High'], self.data['Low'], self.data['Close'], timeperiod), name='DX')
        self.data = self.data.join(dx)
        return self.data

    def DEMA(self, timeperiod: int = 30):
        """Double Exponential Moving Average"""
        dema = pd.Series(talib.DEMA(self.data['Close'], timeperiod), name='DEMA')
        self.data = self.data.join(dema)
        return self.data

    def EMA(self, timeperiod: int = 30):
        """Exponential Moving Average"""
        ema = pd.Series(talib.EMA(self.data['Close'], timeperiod), name='EMA')
        self.data = self.data.join(ema)
        return self.data

    def HT_TRENDLINE(self):
        """Hilbert Transform - Instantaneous Trendline"""
        HT_TRENDLINE = pd.Series(talib.HT_TRENDLINE(self.data['Close']), name='HT_TRENDLINE')
        self.data = self.data.join(HT_TRENDLINE)
        return self.data

    def KAMA(self, timeperiod: int = 30):
        """Kaufman Adaptive Moving Average"""
        kama = pd.Series(talib.KAMA(self.data['Close'], timeperiod), name='KAMA')
        self.data = self.data.join(kama)
        return self.data

    def MA(self, timeperiod: int = 30, matype: int = 0):
        """Moving average"""
        ma = pd.Series(talib.MA(self.data['Close'], timeperiod, matype), name='MA')
        self.data = self.data.join(ma)
        return self.data

    def MIDPOINT(self, timeperiod: int = 14):
        """MidPoint over period"""
        midpoin = pd.Series(talib.MIDPOINT(self.data['Close'], timeperiod), name="MIDPOINT")
        self.data = self.data.join(midpoin)
        return self.data

    def MIDPRICE(self, timeperiod: int = 14):
        """Midpoint Price over period"""
        midprice = pd.Series(talib.MIDPRICE(self.data['High'], self.data['Low'], timeperiod), name='MIDPRICE')
        self.data = self.data.join(midprice)
        return self.data

    def SAR(self, acceleration: int = 0, maximum: int = 0):
        """Parabolic SAR"""
        sar = pd.Series(talib.SAR(self.data['High'], self.data['Low'], acceleration, maximum), name='SAR')
        self.data = self.data.join(sar)
        return self.data

    def SAREXT(self, startvalue: int = 0, offsetonreverse: int = 0, accelerationinitlong: int = 0, accelerationlong: int = 0, accelerationmaxlong: int = 0,
               accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0):
        """Parabolic SAR - Extended"""
        sarext = pd.Series(
            talib.SAREXT(self.data['High'], self.data['Low'], startvalue, offsetonreverse, accelerationinitlong,
                         accelerationlong, accelerationmaxlong, accelerationinitshort, accelerationshort,
                         accelerationmaxshort), name='SAREXT')
        self.data = self.data.join(sarext)
        return self.data

    def T3(self, timeperiod: int = 5, vfactor: int = 0):
        """Triple Exponential Moving Average (T3)"""
        t3 = pd.Series(talib.T3(self.data['Close'], timeperiod, vfactor), name="T3")
        self.data = self.data.join(t3)
        return self.data

    def TEMA(self, timeperiod: int = 30):
        """Triple Exponential Moving Average"""
        tema = pd.Series(talib.TEMA(self.data['Close'], timeperiod), name="TEMA")
        self.data = self.data.join(tema)
        return self.data

    def TRIMA(self, timeperiod: int = 30):
        """Triangular Moving Average"""
        trima = pd.Series(talib.TRIMA(self.data['Close'], timeperiod), name="TRIMA")
        self.data = self.data.join(trima)
        return self.data

    def WMA(self, timeperiod: int = 30):
        """Weighted Moving Average"""
        wma = pd.Series(talib.WMA(self.data['Close'], timeperiod), name="WMA")
        self.data = self.data.join(wma)
        return self.data

    def TRANGE(self):
        """True Range"""
        trange = pd.Series(talib.TRANGE(self.data['High'], self.data['Low'], self.data['Close']), name="TRANGE")
        self.data = self.data.join(trange)
        return self.data