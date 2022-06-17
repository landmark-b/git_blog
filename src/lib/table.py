# The MIT License (MIT)
# Copyright (c) 2022 Yutaro Kida <yut.dev.private@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
# THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.preprocessing import LabelEncoder 

# TODO assume to add ItemCf
class ItemCf():
    """ Item based Collaborative Filter.
    reference: https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-1-knn-item-based-collaborative-filtering-637969614ea
    """

    def __init__(self):
        pass

### General
# 外れ値除去関数
def remove_outlier(df, z_thresh=10, ret_id=False):
    """ Zスコアを用いて外れ値を除去

    Args:
        df (_type_): _description_
        z_thresh (int, optional): _description_. Defaults to 10.
        ret_id (bool, optional): if True, return index. Defaults to False.

    Returns:
        Union(DF, indexes): 
    """
    if ret_id:
        ret = df[(np.abs(stats.zscore(df)) < z_thresh).all(axis=1)].index
    else:
        ret = df[(np.abs(stats.zscore(df)) < z_thresh).all(axis=1)]
    return ret

### Visualize

### Geometry
def dms2dd(s):
    """ change DMS lat/lon to DEG format
    NOTE:
        https://stackoverflow.com/questions/50193159/converting-pandas-data-frame-with-degree-minute-second-dms-coordinates-to-deci
    EXAMPLE:
            s = 0°51'56.29"S
    """
    degrees, minutes, seconds, direction = re.split('[°\'"]+', s)
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60);
    direction = direction.strip()
    if direction in ('S','W'):
        dd*= -1
    return dd 


### 
def add_sub_categories(input, main_category:str, sub_categories:list, sep='-'):
    """ カテゴリカル変数を複数のサブカテゴリへ分割する
    Args:
        input (DataFrame): 入力
        main_category (str): 分割対象のカラム名
        sub_categories (list): 分割後のカラム名
        sep (str, optional): 区切り文字. Defaults to '-'.

    Returns:
        [DataFrame]: 分割後のカラムを追加したDataFrame
    """
    buf = input[main_category].str.split(sep, expand=True)
    buf.columns = sub_categories
    input = input.join(buf)

    return input

def apply_le(model, series:pd.Series):
    """ LabelEncoderの適用
    Args:
        model : LabelEncoder インスタンス
        series (pd.Series): 変換対象のラベル

    Returns:
        [list]: 変換後の値

    Example:
        model = LabelEncoder()
        labels = pd.concat([train['col'], test['col']]).unique() # エンコード対象の全要素を指定
        model.fit(labels)
        df_data['main_category'] = apply_le(df_data['main_category'])
    """
    encoded_label = model.transform(series)

    return encoded_label


def calc_missing_rate(df: pd.DataFrame):
    """ データフレームの列毎の欠損率を降順で出力 """
    print((df.isnull().sum()/len(df)).sort_values(ascending=False))

### Time Series
def insert_datetime_day(df, loc=1, loc_dow=5):
    """ insert 'datetime' col and dayofweek using year, month, day cols.
    Args:
        loc: insert col location for datetime. if None, not insert.
        loc_dow: insert col location for dayofweek. if None, not insert.
    """
    if loc != None and 'datetime' not in df.keys():
        df.insert(loc, 'datetime', pd.to_datetime(df[['year', 'month', 'day']]))
    
    if loc_dow != None and 'dayofweek' not in df.keys():
        df.insert(loc_dow, 'dayofweek', df['datetime'].dt.dayofweek)

    return df

def insert_datetime_day(df, loc=1, loc_dow=5):
    """ insert 'datetime' col and dayofweek using year, month, day cols.
    Args:
        loc: insert col location for datetime. if None, not insert.
        loc_dow: insert col location for dayofweek. if None, not insert.
    """
    if loc != None and 'datetime' not in df.keys():
        df.insert(loc, 'datetime', pd.to_datetime(df[['year', 'month', 'day']]))
    
    if loc_dow != None and 'dayofweek' not in df.keys():
        df.insert(loc_dow, 'dayofweek', df['datetime'].dt.dayofweek)

    return df


