import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

class correlation:
    @staticmethod
    def compute_correlation(data, method: str = "spearman", isShow: bool = True):
        # get correlations of each features in dataset
        corrmat = data.corr(method).abs()
        if isShow:
            top_corr_features = corrmat.index
            plt.figure(figsize=(100, 100))
            # plot heat map
            g = sns.heatmap(data[top_corr_features].corr(method), annot=True, cmap="RdYlGn")
        return corrmat
    @staticmethod
    def reducing_dimension(cor_matrix, th: int = 0.75):
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > th)]
        return to_drop

if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("/Users/admin/PycharmProjects/Market_Prediction_Using_News_and_Indicators/Crypto_Currencies_Data/BTC_indicators.csv")
    df.drop(columns="time", inplace=True)
    corr_mat = correlation.compute_correlation(df)
    red = correlation.reducing_dimension(corr_mat)