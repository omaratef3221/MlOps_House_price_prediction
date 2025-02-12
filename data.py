import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import json

def get_cleaned_dataset(path = "perth-house-prices/all_perth_310121.csv", encoder_path = ".", args = None):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df.drop(["ADDRESS", 
         "NEAREST_STN", 
         "DATE_SOLD", 
         "POSTCODE", 
         "LATITUDE", 
         "LATITUDE", 
         "NEAREST_SCH", 
         "NEAREST_SCH_RANK"], 
        inplace = True, 
        axis = 1)
    Encoder = LabelEncoder()
    df["SUBURB"] = Encoder.fit_transform(df["SUBURB"])
    np.save(os.path.join(encoder_path, "SUBURB_encoder.npy"), Encoder.classes_)
    if args.remove_outliers:
        remove_outliers(df, "BUILD_YEAR")
        remove_outliers(df, "LAND_AREA")
        remove_outliers(df, "NEAREST_STN_DIST")
        remove_outliers(df, "CBD_DIST")
        remove_outliers(df, "NEAREST_SCH_DIST")

    return df

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.3)
    Q3 = df[column].quantile(0.7)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR

    upper_array = np.where(df[column] >= upper)[0]
    lower_array = np.where(df[column] <= lower)[0]
    try:
        df.drop(df.index[upper_array], inplace =True)
    except:
        pass
    try:
        df.drop(df.index[lower_array], inplace = True)
    except:
        pass