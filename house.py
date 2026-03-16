import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# dataset load
data = pd.read_csv("python\house_pridict_app\house_data.csv")

X = data[['area','bedrooms','age']]
y = data['price']

# split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# model train
model = LinearRegression()
model.fit(X_train,y_train)

# model save
joblib.dump(model,"house.pkl")

print("Model saved successfully!")