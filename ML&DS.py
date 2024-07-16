#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc


# In[2]:


df = pd.read_csv('Fraud.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df['isFraud'].describe()


# In[6]:


# Assuming your DataFrame is named 'df'
null_counts = df.isnull().sum()

# Display the null value counts
print(null_counts)


# In[7]:


# Check for missing values
missing_values = df.isnull().sum()

# Print the columns with missing values
print(missing_values[missing_values > 0])

# Handling missing values in numeric columns by filling with median
numeric_columns = df.select_dtypes(include='number').columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Handling missing values in non-numeric columns (example: filling with mode)
non_numeric_columns = df.select_dtypes(exclude='number').columns
for column in non_numeric_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Verify if missing values are handled
print(df.isnull().sum())


# In[8]:


import numpy as np

# Function to detect outliers using the IQR method
def detect_outliers(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[col] < lower_bound) | (data[col] > upper_bound)]

# Detecting and handling outliers for all numeric columns
for col in numeric_columns:
    outliers = detect_outliers(df, col)
    print(f"Outliers detected in {col}:")
    print(outliers)
    
    # Handling outliers by capping
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])


# In[9]:


# Assuming your DataFrame is named 'df'
count_nameOrig_starts_with_M = df[df['nameDest'].str.startswith('M')].shape[0]

# Display the count of samples where nameDest starts with 'M'
print(count_nameOrig_starts_with_M)


# In[10]:


# Assuming your DataFrame is named 'df'
count_nameOrig_starts_with_M = df[df['nameOrig'].str.startswith('M')].shape[0]

# Display the count of samples where nameOrig starts with 'M'
print(count_nameOrig_starts_with_M)


# In[11]:


# Assuming your DataFrame is named 'df'
def label_encode_nameDest(nameDest):
    if nameDest.startswith('M'):
        return 0
    elif nameDest.startswith('C'):
        return 1
    else:
        return None  # Handle other cases if necessary

df['nameDest_encoded'] = df['nameDest'].apply(label_encode_nameDest)
# Display the updated DataFrame
print(df[['nameDest', 'nameDest_encoded']])


# In[12]:


from sklearn.preprocessing import LabelEncoder

# Assuming your DataFrame is named 'df'
le = LabelEncoder()

df['type_encoded'] = le.fit_transform(df['type'])

# Display the updated DataFrame
print(df[['type', 'type_encoded']])


# In[13]:


df.head()


# In[14]:


df.columns.tolist()


# In[15]:


features = ['step','type_encoded', 'amount', 'nameDest_encoded', 'oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFlaggedFraud']
target = 'isFraud'

X = df[features]
y = df[target]


# In[16]:


X.head()


# In[17]:


# Feature Selection
X = df.drop('isFraud', axis=1)
y = df['isFraud']


# In[19]:


# Preprocessing Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=[object]).columns.tolist()

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# In[20]:


# Create a pipeline that first preprocesses the data and then fits the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


# In[21]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


# Fit the model
pipeline.fit(X_train, y_train)


# In[23]:


# Model Evaluation
y_pred = pipeline.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[24]:


# Check the output shape of predict_proba
y_pred_prob = pipeline.predict_proba(X_test)
print("Shape of predict_proba output:", y_pred_prob.shape)

if y_pred_prob.shape[1] == 2:
    # ROC Curve
    y_pred_prob = y_pred_prob[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
else:
    print("The classifier did not return probability estimates for both classes.")


# In[25]:


# Key Factors Identification
model = pipeline.named_steps['classifier']
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]


# In[26]:


# Extract feature names after preprocessing
feature_names = numeric_features + list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features))


# In[27]:


for f in range(X.shape[1]):
    print(f"{feature_names[indices[f]]}: {importances[indices[f]]}")


# In[28]:


# Infrastructure Update Suggestions
def prevention_measures():
    measures = [
        "Implement multi-factor authentication.",
        "Regularly update and patch systems.",
        "Monitor transactions in real-time.",
        "Use encryption for sensitive data.",
        "Conduct regular security audits."
    ]
    return measures

print(prevention_measures())


# In[29]:


# Implementation Evaluation
def evaluate_implementation():
    # Suggest metrics to evaluate the implemented actions
    metrics = [
        "Reduction in the number of fraudulent transactions.",
        "Improvement in detection rate of fraudulent transactions.",
        "Decrease in false positive rate.",
        "User feedback on system changes.",
        "Regular security audit results."
    ]
    return metrics

print(evaluate_implementation())


# In[ ]:




