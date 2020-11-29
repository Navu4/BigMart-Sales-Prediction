import numpy as np
import pandas as pd
import sklearn
import xgboost
import pickle
#Import mode function:
from scipy.stats import mode


#Read files:
train = pd.read_csv("train.csv")

df = train.copy()

# Handling Missing values
#We found two variables with missing values – Item_Weight and Outlet_Size
#### Replacing the missing values of weight with the average weight of the same product
def impute_Item_Weight(df):
    """ This Function Replace the missing values of the Item_weight
    """
    # #Determine the average weight per item:
    item_avg_weight = df.groupby(["Item_Identifier"])["Item_Weight"].mean()
    item_avg_weight

    #Get a boolean variable specifying missing Item_Weight values
    miss_bool = df['Item_Weight'].isnull() 

    #Impute data and check #missing values before and after imputation to confirm
    print('Orignal #missing: %d'% sum(miss_bool))
    df.loc[miss_bool,'Item_Weight'] = df.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight.loc[x])
    print('Final #missing: %d'% sum(df['Item_Weight'].isnull()))
    
    return df

### Lets impute Outlet_Size with the mode of the Outlet_Size for the particular type of outlet.
def impute_Outlet_size(df):
    """ This function replace the missing Outlet_size values
    """
    #Determing the mode for each
    outlet_size_mode = df.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x:mode(x).mode[0]) )
    print('Mode for each Outlet_Type:')
    print(outlet_size_mode)

    #Get a boolean variable specifying missing Item_Weight values
    miss_bool = df['Outlet_Size'].isnull() 

    #Impute data and check #missing values before and after imputation to confirm
    print('\nOrignal #missing: %d'% sum(miss_bool))
    df.loc[miss_bool,'Outlet_Size'] = df.loc[miss_bool,'Outlet_Type'].apply(lambda x: outlet_size_mode[x])
    print('\nFinal #missing: %d'%sum(df['Outlet_Size'].isnull()))
    
    return df

def Handling_missing_values(df):
    """ Handling the missing values Main Function
    """
    df = impute_Item_Weight(df)
    df = impute_Outlet_size(df)
    
    miss_after_bool = df['Item_Weight'].isnull() 
    df = df.loc[~miss_after_bool,:]
    
    return df    

# Handling the Categorical Data

### Modify Item_Visibility
## We noticed that the minimum value here is 0, which makes no practical sense.    
def modify_item_visibility(df):
    #Determine average visibility of a product
    visibility_avg = df.pivot_table(values='Item_Visibility', index='Item_Identifier')

    #Impute 0 values with mean visibility of that product:
    miss_bool = (df['Item_Visibility'] == 0)

    print('Number of 0 values initially: %d'%sum(miss_bool))
    df.loc[miss_bool,'Item_Visibility'] = df.loc[miss_bool,'Item_Identifier'].apply(lambda x: visibility_avg.loc[x])
    
    return df

# We can clearly observe that the First 2 characters of the Item ID is same for the One kind of Item Type. Example: DR is the code for Soft Drinks, NC is the code of Non- Consumable Products and FD is for Food products

### Create a broad category of Type of Item
def broad_item_type(df):
    #Get the first two characters of ID:
    df['Item_Type_Combined'] = df['Item_Identifier'].apply(lambda x: x[0:2])
    #Rename them to more intuitive categories:
    df['Item_Type_Combined'] = df['Item_Type_Combined'].map({'FD':'Food',
                                                                 'NC':'Non-Consumable',
                                                                 'DR':'Drinks'})
    
    return df


def cal_outlet_year(df):
    """ Calculating the Outlet Year
    """
    #Years:
    df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']
    
    return df

def modify_item_fat_content(df):
    #Change categories of low fat:
    print('Original Categories:')
    print(df['Item_Fat_Content'].value_counts())

    print('\nModified Categories:')
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat',
                                                                 'reg':'Regular',
                                                                 'low fat':'Low Fat'})
    print(df['Item_Fat_Content'].value_counts())
    
    return df

def non_consumable_category(df):
    #Mark non-consumables as separate category in low_fat:
    df.loc[df['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"
    print(df['Item_Fat_Content'].value_counts())
    
    return df

def Item_Visibility_MeanRatio(data):
    #Get all Item_Visibility mean values for respective Item_Identifier
    visibility_item_avg = data.pivot_table(values='Item_Visibility',index='Item_Identifier')

    func = lambda x: x['Item_Visibility']/visibility_item_avg['Item_Visibility'][visibility_item_avg.index == x['Item_Identifier']][0]
    data['Item_Visibility'] = data.apply(func,axis=1).astype(float)
    data['Item_Visibility'].describe()
    
    return data

def Handling_categorical_data(df):
    
    # Modify Item_Visibility
    df = modify_item_visibility(df)
    
    # Create a broad category of Type of Item
    df = broad_item_type(df)
    
    # Calculating the Outlet Year
    df = cal_outlet_year(df)
    
    # Modify categories of Item_Fat_Content
    df = modify_item_fat_content(df)
    
    # Mark non-consumables as separate category in low_fat
    df = non_consumable_category(df)
    
    df = Item_Visibility_MeanRatio(df)
    
    return df 

#Import library:
# from sklearn.preprocessing import LabelEncoder

# def label_encoding(df):
#     le = LabelEncoder()
#     #New variable for outlet
#     df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
#     df['Outlet']
#     var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
#     le = LabelEncoder()
#     for i in var_mod
#         df[i] = le.fit_transform(df[i])
        
#     return df    

def One_hot_encoding(df):
    #One Hot Coding:
    df = pd.get_dummies(df, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'],drop_first = True)
    
    return df

def Encoding(df):
    
    # Label encoding
#     df = label_encoding(df)
    
    # #One Hot Coding:
    df = One_hot_encoding(df)
    
    #Created a list remove_cols to remove those columns which doesn't required for Model Building
    remove_cols = [
        'Item_Identifier',
        'Item_Type',
        'Outlet_Identifier',
        'Outlet_Establishment_Year'
    ]
    df = df.drop(remove_cols,axis =1)
    
    return df    

def Data_preprocessing(df):
    
    # Handling Missing Values
    df = Handling_missing_values(df)
    
    # Handling Categorical Data
    df = Handling_categorical_data(df)
    
    # Label and One Hot Encoding
    df = Encoding(df)

    return df

## Data PreProcessing 
#### Handling Missing Values
#### Handling Categorical Data 
#### Label and One Hot Encoding
df = Data_preprocessing(df)


#Export files as modified versions:
df.to_csv("train_modified.csv",index=False)


y = df.Item_Outlet_Sales.values
X = df.drop('Item_Outlet_Sales',axis = 1)

# Parameter using Hyperparameter Optimisation
model = xgboost.XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
            colsample_bynode=1, colsample_bytree=0.4, gamma=0.0, gpu_id=-1,
            importance_type='gain', interaction_constraints='',
            learning_rate=0.25, max_delta_step=0, max_depth=15,
            min_child_weight=1, monotone_constraints='()',
            n_estimators=100, n_jobs=0, num_parallel_tree=1, random_state=0,
            reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
            tree_method='exact', validate_parameters=1, verbosity=None)
# Fitting the model 
model.fit(X,y)

filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))
