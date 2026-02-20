import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

data = {
    "Hours":[1,2,3,4,5,6,7,8,9,10],
    "Sleep":[6,7,5,8,6,7,5,8,6,7],
    "Attendance":[60,65,70,75,80,85,90,92,95,98],
    "Score":[30,40,50,55,65,72,80,88,92,96]
}

df = pd.DataFrame(data)

X = df[["Hours","Sleep","Attendance"]]
y = df["Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl","wb"))

print("Model Trained and Saved Successfully!")