import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder 


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

def apply_le(series:pd.Series, labels=[]):
    """ LabelEncoderの適用
    Args:
        input (pd.Series): 変換対象のラベル
        labels (list, optional): ラベルの種類。指定しない場合はinputから生成する。

    Returns:
        [list]: 変換後の値
    """
    le = LabelEncoder()

    if not labels:
        labels = series.unique()

    le.fit(labels)
    encoded_label = le.transform(series)

    return encoded_label

