import numpy as np
import boto3
import numpy as np
import pandas as pd
from utils import get_args_parser
from remasker_impute import ReMasker
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

X_train = pd.read_csv('sample_train.csv',header=None)


print(X_train.shape)

print("fitting started")
# Initialize your imputer
imputer = ReMasker()

# Since ReMasker is an imputer, we directly fit it on X_train
imputer.fit(X_train, save_path='MAE_test')

print("fitting finished")

