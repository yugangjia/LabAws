
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

X_test = pd.read_csv('sample_test.csv')
X_train = pd.read_csv('sample_train.csv')
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
imputer.fit(X_train)

print("fitting finished")

# Function to mask values in a column
def mask_values(data, column):
    masked_data = data.copy()
    # Randomly mask a certain percentage of the column - adjust as necessary
    mask = np.random.rand(len(masked_data)) < 0.1
    masked_data.loc[mask, column] = np.nan
    return masked_data
print(X_test.columns)
imputedResult=pd.DataFrame()
originalTest=pd.DataFrame()
# Evaluate performance for each of the testX columns
for column, column_name in enumerate(X_test.columns):
    print("we are in")
    # Create a copy of X_test with the current column masked
    column=column
    X_test_masked = mask_values(X_test, column_name)
    
    # Impute missing values
    X_test_imputed =  pd.DataFrame(imputer.transform(X_test_masked).cpu().numpy())
    

    thingOne=X_test_imputed.iloc[:,column].dropna()
    imputedResult = pd.concat([imputedResult, thingOne], axis=1)
    thingTwo=X_test.iloc[:,column].dropna()

    originalTest=pd.concat([originalTest, thingTwo], axis=1)

    print("imputedResult")
    print(imputedResult.columns)
    print("originalTest")
    print(originalTest.columns)

imputedResult.to_csv("imputedResult.csv")
originalTest.to_csv("originalTest.csv")
with open('evaluation_results_yesterday.txt', 'w') as file:
    for column, column_name in enumerate(X_test.columns[:8]):
        # Create a copy of X_test with the current column masked
        X_test_masked = X_test.copy()
        X_test_masked.iloc[:,column]=np.nan
        
        # Impute missing values
        X_test_imputed =  pd.DataFrame(imputer.transform(X_test_masked).cpu().numpy())
        


        # Calculate RMSE, MAE, and R2 for the imputed column
        rmse = sqrt(mean_squared_error(X_test.iloc[:, column].dropna(), X_test_imputed.iloc[:, column].dropna()))
        mae = mean_absolute_error(X_test.iloc[:, column].dropna(), X_test_imputed.iloc[:, column].dropna())
        r2 = r2_score(X_test.iloc[:, column].dropna(), X_test_imputed.iloc[:, column].dropna())
        
        # Construct the output string
        output_str = f"Evaluation for {column_name}: RMSE = {rmse}, MAE = {mae}, R2 = {r2}\n"
        
        # Write to file and print
        file.write(output_str)
        print(output_str)