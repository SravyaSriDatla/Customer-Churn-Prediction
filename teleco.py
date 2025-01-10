import numpy as np
import pandas as pd
import pickle

df = pd.read_csv(r'/Users/bannusagi/Downloads/Customer_churn_project/Telco-Customer-Churn.csv')
df.info()

features = ['gender','SeniorCitizen', 'tenure' ,'Partner', 'Dependents', 'PhoneService','MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup','DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies','Contract', 'PaperlessBilling', 'PaymentMethod',  'Churn']

df1 = df[features]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1 = df1.apply(le.fit_transform)

X = df1.iloc[:,:-1]
y = df1.iloc[:,-1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state=42, n_estimators=10, max_depth=7)
classifier.fit(X_train, y_train)
classifier.predict(X_test)

print(classifier.score(X_test,y_test))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, classifier.predict(X_test))
cm

bias = classifier.score(X_train,y_train)
print(bias)

varience = classifier.score(X_test,y_test)
print(varience)

# Save the trained model to disk
filename = 'TelcoChurn.pkl'
with open(filename, 'wb') as file:
    pickle.dump(classifier, file)