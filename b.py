
import torch
import pandas as pd
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


X_train = pd.read_csv('X_train.csv',header=None)
print(X_train.head)
print(X_train.shape)
# # Function to generate new column names
# def generate_new_name(old_name):
#     counter=1
#     mapping = {}
#     # Check if the column name is numeric or has a "_yesterday" suffix
#     base_name = old_name.replace('_yesterday', '')
#     if base_name.isdigit():
#         if base_name not in mapping:
#             # Assign a new testX name and increment the counter
#             mapping[base_name] = f'test{counter}'
#             counter += 1
#         # Return the mapped name, with "_yesterday" suffix if necessary
#         return mapping[base_name] + ('_yesterday' if '_yesterday' in old_name else '')
#     else:
#         # Non-numeric columns remain unchanged
#         return old_name

# # Apply the renaming function to each column
# test.columns = [generate_new_name(col) for col in test.columns]
# X_test = test.filter(regex='^test[1-8](_yesterday)?$') 

# train.columns = [generate_new_name(col) for col in train.columns]
# X_train = train.filter(regex='^test[1-8](_yesterday)?$') 

print("fitting started")
# Initialize your imputer
imputer = ReMasker()

# Since ReMasker is an imputer, we directly fit it on X_train
imputer.fit(X_train,'MAE_10lab')

print("fitting finished")

# Function to mask values in a column

print(X_test.columns)

ItemID = 0

y_test = X_test.iloc[:,ItemID]
y_last = X_test.iloc[:,20+ItemID]
print("imputation started")
X_test_masked = X_test.copy()
X_test_masked.iloc[:,ItemID] = np.nan
y_pred =  imputer.transform(X_test_masked).cpu().numpy()
print("imputation done")


np.savetxt('y_test.csv', y_test, delimiter=',')
np.savetxt('y_last.csv', y_last, delimiter=',')
np.savetxt('y_pred.csv', y_pred, delimiter=',')

print("saves done")
normal_ranges = [(3.5,5),(136,145),(98,106),(37,50),(0.7,1.3),(23,28),(7,13),(8,20),(0,200),(150,450)]

mse, r2, cm = mse_r2_confusion_matrix(y_test, y_pred, normal_ranges[ItemID])

print("Overall MAE MSE:", round(mse,3))
print("Overall MAE R-squared:", round(r2,3))
print("Overall MAE Confusion matrix:\n", cm)


valid_pos = ~np.isnan(y_last)
print('Test with the last value',len(y_last[valid_pos]))
mse, r2, cm = mse_r2_confusion_matrix(y_test, y_last, normal_ranges[ItemID])
print("last value prediction MSE per column:", round(mse,3))
print("last value prediction R-squared per column:", round(r2,3))
print("last value prediction Confusion matrix:\n", cm)

mse, r2, cm = mse_r2_confusion_matrix(y_test[valid_pos], y_pred[valid_pos], normal_ranges[ItemID])
print("MAE for those with last value MSE :", round(mse,3))
print("MAE for those with last value R-squared:", round(r2,3))
print("MAE for those with last value Confusion matrix:\n", cm)