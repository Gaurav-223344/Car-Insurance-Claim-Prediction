from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import joblib
import pandas as pd
import numpy as np
from pipeline import FeatureHandlingForApp

app = Flask(__name__)
model = joblib.load('RFmodel.joblib')
fha = FeatureHandlingForApp()

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        age = str(request.form['age'])
        driving_experiance = str(request.form['driving_experiance'])
        income = str(request.form['income'])
        credit_score = float(request.form['credit_score'])
        mileage = int(request.form['mileage'])
        speeding_violations = int(request.form['speeding_violations'])
        past_accedents = int(request.form['past_accedents'])
        vehical_ownership = int(request.form['vehical_ownership'])
        married = int(request.form['married'])
        children = int(request.form['children'])
        before = int(request.form['before'])
        
        
        data = [age,driving_experiance,income,credit_score,vehical_ownership,married,children,mileage,speeding_violations,past_accedents,before]
        df = pd.DataFrame(data).transpose()
        df.columns = [['AGE', 'DRIVING_EXPERIENCE', 'INCOME', 'CREDIT_SCORE', 'VEHICLE_OWNERSHIP', 'MARRIED', 'CHILDREN', 'ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS', 'PAST_ACCIDENTS', 'VEHICLE_YEAR_before 2015']] 
        
        train_df = pd.read_csv("x_train.csv")
        df_new = fha.transform(df,train_df)
        prediction = model.predict(df_new)

        if prediction[0]==0:
            return render_template('index.html',prediction_texts="Customer will not claim for car insurance")
        else:
            return render_template('index.html',prediction_texts="Customer will claim for car insurance")


    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
    # app.debug = True
    # app.run()