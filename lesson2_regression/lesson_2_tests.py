import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Sample data
dataset_partial = pd.DataFrame({
    'age': [25, 30, 35],
    'income': [50000, 60000, 75000],
    'city': ['New York', 'London', 'Paris'],
    'color': ['red', 'blue', 'green']
})

# Define columns
numerical_columns = ['age', 'income']
categorical_columns = ['city', 'color']

# Numerical pipeline
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ('one_hot_encoder', OneHotEncoder( handle_unknown='ignore')),
])

# Full pipeline with ColumnTransformer
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, numerical_columns),
    ("cat", cat_pipeline, categorical_columns),
])

# Transform data
dataset_prepared = full_pipeline.fit_transform(dataset_partial)

# Convert to DataFrame (optional)
feature_names = (numerical_columns + 
                full_pipeline.named_transformers_['cat']
                .named_steps['one_hot_encoder']
                .get_feature_names_out(categorical_columns).tolist())
dataset_prepared_df = pd.DataFrame(dataset_prepared, columns=feature_names)

print(dataset_prepared)
