import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('Housing.csv')

categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                    'airconditioning', 'prefarea', 'furnishingstatus']

for col in categorical_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0, 'furnished': 1, 'semi-furnished': 2, 'unfurnished': 3})

X = df.drop('price', axis=1)
y = df['price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

with open('housing_app/housing_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('housing_app/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
