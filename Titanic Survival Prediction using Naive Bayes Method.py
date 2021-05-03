#importing libraries

import numpy as np
import pandas as pd

#Loading dataset

dataset = pd.read_csv('titanicsurvival.csv')

#summarizing dataset

print(dataset.shape)
print(dataset.head(5))

#mapping text data into Binary Values

income_set = set(dataset['Sex'])
dataset['Sex'] = dataset['Sex'].map({'female':0, 'male':1}).astype(int)
print(dataset.head)

#segregating dataset into X and Y
#X(Input/Independent Variable) and Y(Output/Dependent Variable)

X = dataset.drop('Survived',axis='columns')
X
Y = dataset.Survived
Y

#finding NA values from X

X.columns[X.isna().any()]

#removing NA values from X

X.Age = X.Age.fillna(X.Age.mean())

#testing again for any na value

X.columns[X.isna().any()]

#splitting dataset into Train and Test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25,random_state =0)

#training

from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB
model1 = GaussianNB()
model1.fit(X_train, y_train)

model2 = MultinomialNB()
model2.fit(X_train, y_train)

model3 = ComplementNB()
model3.fit(X_train, y_train)

model4 = BernoulliNB()
model4.fit(X_train, y_train)

model5 = CategoricalNB()
model5.fit(X_train, y_train)


#Predicting, wheather Person Survived or Not

pclassNo = int(input("Enter Person's Pclass number: "))
gender = int(input("Enter Person's Gender 0-female 1-male(0 or 1): "))
age = int(input("Enter Person's Age: "))
fare = float(input("Enter Person's Fare: "))
person = [[pclassNo,gender,age,fare]]
result1 = model1.predict(person)
result2 = model2.predict(person)
result3 = model3.predict(person)
result4 = model4.predict(person)
result5 = model5.predict(person)

print('By Gaussian Naive Bayes Method:')
if result1 == 1:
  print("Person might be Survived")
else:
  print("Person might not be Survived")

print('By Multinomial Naive Bayes Method:')
if result2 == 1:
  print("Person might be Survived")
else:
  print("Person might not be Survived")

print('By Complement Naive Bayes Method:')
if result3 == 1:
  print("Person might be Survived")
else:
  print("Person might not be Survived")

print('By Bernoulli Naive Bayes Method:')
if result4 == 1:
  print("Person might be Survived")
else:
  print("Person might not be Survived")

print('By Categorical Naive Bayes Method:')
if result5 == 1:
  print("Person might be Survived")
else:
  print("Person might not be Survived")


#prediction of all Test data

y_pred1 = model1.predict(X_test)
print(np.column_stack((y_pred1,y_test)))

y_pred2 = model2.predict(X_test)
print(np.column_stack((y_pred2,y_test)))

y_pred3 = model3.predict(X_test)
print(np.column_stack((y_pred3,y_test)))

y_pred4 = model4.predict(X_test)
print(np.column_stack((y_pred4,y_test)))

y_pred5 = model5.predict(X_test)
print(np.column_stack((y_pred5,y_test)))


#Accuracy

from sklearn.metrics import accuracy_score
print("Accuracy of the Model 1(Gaussian Naive Bayes Method): {0}%".format(accuracy_score(y_test, y_pred1)*100))
print("Accuracy of the Model 2(Multinomial Naive Bayes Method): {0}%".format(accuracy_score(y_test, y_pred2)*100))
print("Accuracy of the Model 3(Complement Naive Bayes Method): {0}%".format(accuracy_score(y_test, y_pred3)*100))
print("Accuracy of the Model 4(Bernoulli Naive Bayes Method): {0}%".format(accuracy_score(y_test, y_pred4)*100))
print("Accuracy of the Model 5(Categorical Naive Bayes Method): {0}%".format(accuracy_score(y_test, y_pred5)*100))