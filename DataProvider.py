import numpy as np
import pandas as pd
from pandas import DataFrame

# df: DataFrame = pd.read_csv("C:/Users/Andrew/Desktop/winequality-red.csv")
df: DataFrame = pd.read_excel("C:/Users/Andrew/Desktop/led_exel_p.xlsx", sheet_name="led")
__whole_data_set__: np.ndarray = df.to_numpy()

buildingSet: np.ndarray = __whole_data_set__[:1000, :]
testingSet: np.ndarray = __whole_data_set__[1000:, :]
