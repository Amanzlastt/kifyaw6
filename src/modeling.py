import pandas as pd 
data = pd.read_csv('C:\\Users\\Aman\\Desktop\\kifyaw6\\data\\filtered.csv')


from sklearn.model_selection import train_test_split

x = data.drop('StdDevTransactionAmount', axis=1)
y = data['StdDevTransactionAmount']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

model.fit(x_train, y_train)

import pickle
# Save the trained model to a file
with open('trained_model.pkl', 'wb') as file:
    pickle.dump(model, file)