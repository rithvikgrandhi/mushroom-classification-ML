from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


data = pd.read_csv(r"C:\Users\grand\Desktop\ml pesuio\final project\mushrooms.csv")

mappings = list()

encoder = LabelEncoder()

for column in range(len(data.columns)):
    data[data.columns[column]] = encoder.fit_transform(data[data.columns[column]])
    mappings_dict = {index: label for index, label in enumerate(encoder.classes_)}
    mappings.append(mappings_dict)

print(data)
print(mappings)

y = data['class']
X = data.drop('class', axis=1)
scaler = StandardScaler()

X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)

svm_model = SVC(C=1.0, kernel='rbf')
np.sum(y) / len(y)
svm_model.fit(X_train, y_train)
print(f"Support Vector Machine: {svm_model.score(X_test, y_test)}")