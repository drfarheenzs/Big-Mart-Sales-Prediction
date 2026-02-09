#!/usr/bin/env python
# coding: utf-8

# ## 1 Problem Understanding

# Predict : Item_Outlet_Sales for each Item-Outlet combination in the test dataset.

# ### 1.1 Business Goal

# Big Mart wants to understand:
# 1. Which product attributes increase sales?
# 2. Which outlet characteristics driv higher revenue?
# 3. How to optimize inventory and outlet strategy?

# ### 1.2 Type of Problem

# As per the business problem it can be identified that it is a Supervised Regression Problem because the Target variable is continous (Sales value) and historical labelled data is available.

# ## 2 Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


# ## 3 Data Loading

# In[2]:


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


# ### 3.1 Data Understanding

# In[3]:


train.shape


# In[4]:


train.columns


# In[5]:


test.shape


# In[6]:


test.columns


# In[7]:


test_original = test.copy()


# In[8]:


test_original.columns


# From the above observation it can be noted that Train dataset has 8523 rows and test dataset has 5681 rows

# ## 4 Data Preprocessing

# ### 4.1 Combine Train and Test

# For consistent preprocessing the train and test datasets are combined

# In[9]:


train['source'] = 'train'
test['source'] = 'test'

data = pd.concat([train, test], ignore_index = True)


# ### 4.2 Checking for missing values

# In[10]:


data_missing = data.isnull().sum()
print(data_missing)


# Since it is observed that Item_Weight has missing values it can be imputed with the mean weight per item

# In[11]:


data['Item_Weight'] = data.groupby('Item_Identifier')['Item_Weight']\
                            .transform(lambda x:x.fillna(x.mean()))


# In[12]:


data_missing = data.isnull().sum()
print(data_missing)


# In[13]:


data['Outlet_Size'].value_counts()


# In[14]:


data['Outlet_Type'].value_counts()


# Since Outlet_Size is a categorical value the missing values for the column can be imputed with mode based on outlet type

# In[15]:


data['Outlet_Size'] = data.groupby('Outlet_Type')['Outlet_Size']\
                            .transform(lambda x: x.fillna(x.mode()[0]))


# In[16]:


data_missing = data.isnull().sum()
print(data_missing)


# ### 4.3 Checking for inconsistencies

# In[17]:


data['Item_Fat_Content'].value_counts()


# As it can be observed there are few inconsistencies in the column values for Item_Fat_Content

# In[18]:


data['Item_Fat_Content'].replace({
    'LF': 'Low Fat',
    'low fat': 'Low Fat',
    'reg': 'Regular'
}, inplace = True)
data['Item_Fat_Content'].value_counts()


# In[19]:


data['Item_Visibility'].value_counts()


# Some of the items have zero visibility which is not realistic, hence resolving the inconsistency by imputing with mean value

# In[20]:


data['Item_Visibility'] = data.groupby('Item_Identifier')['Item_Visibility']\
                                .transform(lambda x: x.replace(0, x.mean()))


# In[21]:


data['Item_Type'].value_counts()


# In[22]:


data.columns


# In[23]:


data['Outlet_Location_Type'].value_counts()


# In[24]:


data['Outlet_Type'].value_counts()


# ## 5 Feature Engineering

# ### Outlet Age

# Older outlets may have better sales and can help in prediction

# In[25]:


data['Outlet_Establishment_Year'].value_counts()


# In[26]:


data['Outlet_Age'] = 2013 - data['Outlet_Establishment_Year']
data.head()


# ### 5.2 Item Type from Item Identifier

# The first two letters in item identifier represent category

# In[27]:


data['Item_Category'] = data['Item_Identifier'].str[:2]


# In[28]:


# Mapping the item category
data['Item_Category'].replace({
    'FD': 'Food',
    'DR': 'Drinks',
    'NC': 'Non-Consumable'
}, inplace = True)


# In[29]:


# No-Consumable Fat Content Fix
data.loc[data['Item_Category'] == 'Non-Consumable',
        'Item_Fat_Content'] == 'Non-Edible'


# ### 6 Encoding categorical variables

# In[30]:


source = data['source']
data = data.drop('source', axis = 1)


# In[31]:


data = pd.get_dummies(data, drop_first=True)


# In[32]:


data['source'] = source


# ## 7 Splitting Train and Test Dataset

# Since the source variable was added for both the dataset so the splitting of the data maintains the consistency of the dataset and the percentage in which the dataset was initially divided

# In[34]:


data.head()


# In[35]:


train = data[data['source']=='train']
test = data[data['source']=='test']

train.drop(['source'], axis = 1, inplace=True)
test.drop(['source','Item_Outlet_Sales'], axis=1, inplace=True)


# ### 7.1 Defining Target variable (y)

# In[36]:


y = train['Item_Outlet_Sales']


# ### 7.2 Defining Features (X)

# In[37]:


X = train.drop('Item_Outlet_Sales', axis=1)


# ### 7.3 Preparing Test Features

# In[38]:


# Test data does not have salec column
X_test = test.drop('Item_Outlet_Sales', axis=1, errors='ignore') #errors=ignore prevents error if column doesnt exist


# ### 7.4 Creating Validation Dataset

# In[39]:


X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ## 8 Model Building

# I am starting with simple model and gradually improving to get the best prediction for sales

# ### 8.1 Linear Regression

# In[40]:


lr = LinearRegression()
lr.fit(X_train, y_train)


# #### 8.1.1 Ridge Regression

# In[41]:


ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)


# ### 8.2 Random Forest

# In[42]:


rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42
)
rf.fit(X_train, y_train)


# ### 8.3 XGBoost

# In[43]:


xgb = XGBRegressor(
        learning_rate=0.05,
        n_estimators=500,
        max_depth=6
)
xgb.fit(X_train, y_train)


# ## 9 Model Evaluation

# In[44]:


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ### 9.1 Linear Regression

# In[45]:


# Using Root Mean Squared error for model evaluation
pred_lr = lr.predict(X_valid)

rmse_lr = rmse(y_valid, pred_lr)
print("Linear Regression RMSE:", rmse_lr)


# In[46]:


pred_lr.min(), pred_lr.max()


# #### 9.1.1 Ridge Regression

# In[47]:


pred_ridge = ridge.predict(X_valid)

rmse_ridge = rmse(y_valid, pred_ridge)
print(rmse_ridge)


# ### 9.2 Random Forest

# In[48]:


pred_rf = rf.predict(X_valid)
rmse_rf = rmse(y_valid, pred_rf)
print("Random Forest RMSE:", rmse_rf)


# ### 9.3 XGBoost

# In[49]:


pred_xgb = xgb.predict(X_valid)

rmse_xgb = rmse(y_valid, pred_xgb)
print("XGBoost RMSE:", rmse_xgb)


# Linear Regression was initially used as a baseline model to establish a reference performance for predicting Item_Outlet_Sales. Linear regression assumes:
# 
# A linear relationship between features and target
# 
# Independence between predictor variables
# 
# Low multicollinearity among features
# 
# Numerical stability in coefficient estimation
# 
# However, these assumptions are not fully satisfied

# #### Issues Observed with Linear Regression

# During model evaluation, Linear Regression produced an extremely high RMSE value compared to tree-based models (Random Forest and XGBoost).
# 
# This occurred because of numerical instability in coefficient estimation, primarily caused by:
# 
# Multicollinearity
# 
# After one-hot encoding categorical variables such as:
# 
# Outlet Type
# 
# Outlet Size
# 
# Outlet Location Type
# 
# Item Type
# Several features becaome highly correlated with each other

# #### High Dimensional Sparse Features

# One-hot encoding increases the number of features, many of which are sparse. Linear regression is sensitive to such feature structures and may overfit or produce extreme predictions.

# #### Difference in Feature Scales

# Features such as Item_MRP, Item_Visibility, and Outlet_Age exist on different numerical scales, which further contributes to instability in coefficient estimation.
# 
# As a result, prediction values became excessively large, leading to an unrealistic RMSE.

# #### Why Tree Based Models Worked Better

# Random Forest and XGBoost performed significantly better because:
# 
# They do not assume linear relationships
# 
# They are not affected by multicollinearity
# 
# They automatically capture feature interactions and nonlinear patterns
# 
# They are robust to feature scaling

# #### Why Ridge Regression Was Introduced

# To stabilize the linear model, Ridge Regression was applied.
# The penalty term:
# 
# Shrinks large coefficient values
# 
# Reduces sensitivity to multicollinearity
# 
# Improves numerical stability
# 
# Prevents coefficient explosion
# 
# By constraining coefficient magnitude, Ridge Regression produces more stable predictions while retaining interpretability of a linear model.

# ## 10 Selecting the Final Model

# #### Random Forest is selected as the final mdel as it achieved the lowest RMSE on validation data compared to Ridge Regression and XGBoost, indicating better generalization performance for this dataset.

# ### 10.1 Train on full training data

# In[50]:


rf.fit(X,y)


# ### 10.2 Predict on Test Data

# In[51]:


final_predictions = rf.predict(X_test)


# In[52]:


print(final_predictions[:10])


# ## 11 Creating Submission File

# In[53]:


submission = pd.DataFrame({
            'Item_Identifier': test_original['Item_Identifier'],
            'Outlet_Identifier': test_original['Outlet_Identifier'],
            'Item_Outlet_Sales': final_predictions
})
submission.head()


# In[54]:


submission.to_csv("bigmart_sales_prediction.csv", index = False)


# ## 12 Final Conclusion

# The objective of predicting product-level sales across BigMart outlets was addressed using multiple regression models after performing data cleaning, feature engineering, and categorical encoding. Linear Regression showed unstable performance due to multicollinearity and was improved using Ridge Regression, while tree-based models demonstrated superior predictive capability. Among all models evaluated, Random Forest achieved the lowest RMSE on validation data, indicating better generalization performance. The final Random Forest model was retrained on the complete training dataset and used to generate sales predictions for the test dataset. The resulting submission file provides reliable sales forecasts that can help BigMart understand key product and outlet characteristics influencing sales performance.

# In[ ]:




