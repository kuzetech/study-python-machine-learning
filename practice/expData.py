import os
import tarfile
from six.moves import urllib
import pandas as pd

DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml/raw/master/"
HOUSING_PATH = "../datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
print(housing.info())

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))
plt.show()

import numpy as np
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

# print(housing["income_cat"].value_counts() / len(housing))

# housing["income_cat"].hist(bins=50, figsize=(20, 15))
# plt.show()

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# print(len(strat_train_set))
# print(len(strat_test_set))

# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

train_copy = strat_train_set.copy()
# train_copy.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#              s=train_copy["population"]/100, label="population",
#              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
#              )
# plt.legend()
# plt.show()

# corr_matrix = train_copy.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

# attributes = ["median_house_value", "median_income", "total_rooms",
#               "housing_median_age"]
# pd.plotting.scatter_matrix(train_copy[attributes], figsize=(12, 8))
# plt.show()

# train_copy.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# plt.show()

# train_copy["rooms_per_household"] = train_copy["total_rooms"]/train_copy["households"]
# train_copy["bedrooms_per_room"] = train_copy["total_bedrooms"]/train_copy["total_rooms"]
# train_copy["population_per_household"] = train_copy["population"]/train_copy["households"]
# corr_matrix = train_copy.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))