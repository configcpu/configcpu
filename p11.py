from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , confusion_matrix
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
x = iris.data
y = iris.target

print("Feature :" , iris.feature_names)
print(x)
print("Labesl " , iris.target_names)
print(y)

x_train , x_test , y_train , y_test = train_test_split(x,y)
classifier = KNeighborsClassifier()
classifier.fit(x_train , y_train)
y_pred = classifier.predict(x_test)
print('confusion matrx')
print(confusion_matrix(y_test, y_pred))
print('Accuracy metics')
print(classification_report(y_test , y_pred))