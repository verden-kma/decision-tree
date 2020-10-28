import pandas as pd
import numpy as np
from pandas import DataFrame

# df: DataFrame = pd.read_csv("C:/Users/Andrew/Desktop/winequality-red.csv")
# df: DataFrame = pd.read_csv("C:/Users/Andrew/Desktop/winequality-processed.csv")
df: DataFrame = pd.read_excel("C:/Users/Andrew/Desktop/led_exel_p.xlsx")
__wholeDataSet: np.ndarray = df.to_numpy()


buildingSet: np.ndarray = __wholeDataSet[:1000, :]
testingSet: np.ndarray = __wholeDataSet[1000:, :]

# buildingSet: np.ndarray = __wholeDataSet[:10, :]
