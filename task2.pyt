import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA


df = pd.read_csv('titanic.csv')


numeric_features = ['Age', 'Fare']  
categorical_features = ['Embarked', 'Sex']  

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))  
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  
])


scaler = StandardScaler()


pca = PCA(n_components=2)



preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', scaler),
    ('pca', pca)
])


X = df.drop('Survived', axis=1)  # Features
y = df['Survived']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



pipeline.fit(X_train)



X_train_transformed = pipeline.transform(X_train)
X_test_transformed = pipeline.transform(X_test)

