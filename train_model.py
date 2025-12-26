import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("House_Rent_Dataset.csv")

X = df[['BHK','Size','Bathroom','Area Type','City',
        'Furnishing Status','Tenant Preferred','Point of Contact']]
y = df['Rent']

cat_cols = ['Area Type','City','Furnishing Status',
            'Tenant Preferred','Point of Contact']
num_cols = ['BHK','Size','Bathroom']

preprocess = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', 'passthrough', num_cols)
])

model = Pipeline([
    ('preprocess', preprocess),
    ('lr', LinearRegression())
])

model.fit(X, y)

# SAVE MODEL
pickle.dump(model, open('rent_model.pkl', 'wb'))

print("✅ rent_model.pkl CREATED SUCCESSFULLY")
