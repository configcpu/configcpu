# Write a program to demonstrate the working of the decision tree based ID3 algorithm.
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load and display the data
data = pd.read_csv("program07.csv")
print("The first 5 values of data is \n", data.head())

# Split data into features and labels
X = data.iloc[:, :-1]
print("\nThe first 5 values of Train data is \n", X.head())
y = data.iloc[:, -1]
print("\nThe first 5 values of Train output is \n", y.head())

# Encode categorical features
le_outlook = LabelEncoder()
X.Outlook = le_outlook.fit_transform(X.Outlook)
le_Temperature = LabelEncoder()
X.Temperature = le_Temperature.fit_transform(X.Temperature)
le_Humidity = LabelEncoder()
X.Humidity = le_Humidity.fit_transform(X.Humidity)
le_Windy = LabelEncoder()
X.Windy = le_Windy.fit_transform(X.Windy)

print("\nNow the Train data is", X.head())

# Encode target variable
le_PlayTennis = LabelEncoder()
y = le_PlayTennis.fit_transform(y)
print("\nNow the Train output is\n", y)

# Train the classifier
classifier = DecisionTreeClassifier()
classifier.fit(X, y)


def labelEncoderForInput(list1):
    list1[0] = le_outlook.transform([list1[0]])[0]
    list1[1] = le_Temperature.transform([list1[1]])[0]
    list1[2] = le_Humidity.transform([list1[2]])[0]
    list1[3] = le_Windy.transform([list1[3]])[0]
    return [list1]


# Prepare input data
inp1 = ["rainy", "cool", "high", "FALSE"]
pred1 = labelEncoderForInput(inp1)

# Predict and inverse transform
y_pred = classifier.predict(pred1)
print(
    "\nfor input {0}, we obtain {1}".format(
        inp1, le_PlayTennis.inverse_transform(y_pred)[0]
    )
)
