import numpy as np
import boto3
import numpy as np
import pandas as pd
from utils import get_args_parser
from remasker_impute import ReMasker
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
X_test = pd.read_csv('sample_test.csv',header=None)
X_test = X_test.iloc[:100,]
print(X_test.head)
print(X_test.shape)

imputer = ReMasker()
# Load the state dictionary into the model
state = torch.load('MAE_test')
print(imputer.max_epochs)
imputer.load_state_dict(state['model_state_dict'])
print("model loaded")

ItemID = 0

y_test = X_test.iloc[:,ItemID]
print("imputation started")
X_test_masked = X_test.copy()
X_test_masked.iloc[:,ItemID] = np.nan
y_pred =  imputer.transform(X_test_masked).cpu().numpy()
print("imputation done")
print(y_pred)


