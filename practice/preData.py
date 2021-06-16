import os
import pandas as pd

def load_housing_data():
    csv_path = os.path.join("../datasets/housing", "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()

import numpy as np
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# strat_train_set_tr = pd.DataFrame(strat_train_set, columns=strat_train_set.columns)
# print(strat_train_set_tr.info())

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# housing.dropna(subset=["total_bedrooms"])    # 去掉整个街区
# housing.drop("total_bedrooms", axis=1)       # 去掉整个属性
# median = housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median)     # 取中位数赋值

# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy="median")  #指定计算中位数
# housing_num = housing.drop("ocean_proximity", axis=1)   # 仅支持全数据数字属性，所以丢弃文本属性
# imputer.fit(housing_num)    # 关联数据集
# X = imputer.transform(housing_num)  # 补充空值
# housing_tr = pd.DataFrame(X, columns=housing_num.columns)   # 转换成Pandas DataFrame格式
# print(housing_tr.info())

# 通过独热编码处理文本类型属性
# from tool.CategoricalEncoder import CategoricalEncoder
# cat_encoder = CategoricalEncoder()
# housing_cat = housing["ocean_proximity"]
# housing_cat_reshaped = housing_cat.values.reshape(-1, 1)
# housing_cat_1hot = cat_encoder.fit_transform(housing_cat_reshaped)
# print(housing_cat_1hot)

# 特征缩放 = 归一化 或 标准化
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from tool.CombinedAttributesAdder import CombinedAttributesAdder
from tool.DataFrameSelector import DataFrameSelector
from sklearn.pipeline import FeatureUnion
from tool.CategoricalEncoder import CategoricalEncoder

housing_num = housing.drop("ocean_proximity", axis=1)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', CategoricalEncoder(encoding="onehot-dense")),
])

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared)





