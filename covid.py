import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import glob
import datetime

# Alternative option to NY enriched
def johns_hopkins_us_daily_reports():
    files = glob.glob("data/csse_daily_reports*.csv")
    li = []
    for filename in files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame.drop(['Lat', 'Long_', 'ISO3', 'UID', 'FIPS', 'Country_Region'], axis=1, inplace=True)
    return frame

# TODO parse enriched data