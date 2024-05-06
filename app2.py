from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import csv
import numpy as np

app = Flask(__name__)

def assign_class(value):
    if value <= 3000:
        return 3000
    elif value <= 5000:
        one = abs(3000-value)
        two = abs(5000-value)
        if(one<two): return 3000
        else:        return 5000
    elif value <= 8000:
        one = abs(5000-value)
        two = abs(8000-value)
        if(one<two): return 5000
        else:        return 8000
    else:
        return value
    

def assign_dist(value):
    if value == 'Close (within 500m)':
        return 1
    elif value == 'Not close':
        return 0
    else:
        return -1
    
def assign_accom(value):
    if value == 'Furnished':
        return 1
    elif value == 'Not Furnished':
        return 0
    else:
        return -1
    
def assign_ac(value):
    if value == 'AC':
        return 1
    elif value == 'Non-AC':
        return 0
    else:
        return -1
    
def assign_food(value):
    if value == 'Yes':
        return 1
    elif value == 'No':
        return 0
    else:
        return -1
    
def assign_food_type(value):
    if value == 'Veg & Non-veg':
        return 1
    elif value == 'Only veg':
        return 0
    else:
        return -1

def train():

    cell_df = pd.read_csv('final_response.csv')
    cell_df.head()

    cell_df = cell_df.dropna()

    cell_df['class'] = cell_df['Approximate rent paid per month'].apply(assign_class)
    cell_df['close'] = cell_df['Distance from target Institution'].apply(assign_dist)
    cell_df['furnished'] = cell_df['Is the accommodation furnished?'].apply(assign_accom)
    cell_df['ac'] = cell_df['Type of accommodation rented'].apply(assign_ac)
    cell_df['food'] = cell_df['Is food facility available?'].apply(assign_food)
    cell_df['nonveg'] = cell_df['Type of food facility'].apply(assign_food_type)

    feature_df = cell_df[['close','furnished','ac','food','nonveg']]

    # Step 3: Split the Data
    #Independent variables
    X = np.asarray(feature_df)
    #dependent
    y = np.asarray(cell_df['class'])

    # Step 4: Choose a Model
    model = DecisionTreeClassifier()
    print(X)
    print(y)

    # Step 5: Train the Model
    model.fit(X, y)
    return model


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    # Step 1: Load the Data
    model = train()

    # Get the input values from the form
    distance = (request.form['Distance from target Institution'])
    furnishing = (request.form['Is the accommodation furnished?'])
    ac = (request.form['Type of accommodation rented'])
    food = (request.form['Is food facility available?'])
    food_type = (request.form['Type of food facility'])
    
    #formatting inputs
    distance = int(assign_dist(distance))
    furnishing = int(assign_accom(furnishing))
    ac = int(assign_ac(ac))
    food = int(assign_food(food))
    food_type = int(assign_food_type(food_type))
    if(food_type == -1): food_type = 0
    
    # Make a prediction using the loaded model
    prediction = model.predict([[distance,furnishing,ac,food,food_type]])
    prediction = int(prediction)

    prediction = str(prediction)
    
    # Render the prediction result template with the prediction
    return render_template('result.html',prediction = prediction)


if __name__ == '__main__':
    app.run(debug=True)



