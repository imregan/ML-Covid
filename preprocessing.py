# Preprocessing Helpers
import pandas as pd
import numpy as np

# fix dates
import re
from datetime import datetime

# check if column name is date
def is_date(colname):
    regex = re.compile('.*/.*/.*')
    if regex.match(colname):
        return True
    else:
        return False

# month/day/year to year-day-month
def fix_datetime(stringtime):
    datetime_obj = datetime.strptime(stringtime, "%m/%d/%y")
    return datetime_obj.date()

# set column strings to datetimes
def set_column_datetimes(df):
    for col in df.columns:
        if is_date(col):
            df.rename(columns={col:fix_datetime(col)}, inplace=True)
