from flask import Flask, render_template, request
import pandas as pd
import pickle

# Load model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    prediction_proba = None

    if request.method == 'POST':
        #  Retrieve input values from the form
        geography = request.form.get('geography')
        gender = request.form.get('gender')
        age = float(request.form.get('age'))
        balance = float(request.form.get('balance'))
        credit_score = float(request.form.get('credit_score'))
        estimated_salary = float(request.form.get('estimated_salary'))
        tenure = int(request.form.get('tenure'))
        num_of_products = int(request.form.get('num_of_products'))
        has_cr_card = int(request.form.get('has_cr_card'))
        is_active_member = int(request.form.get('is_active_member'))

        # Create a pandas DataFrame for the input
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Geography': [geography],
            'Gender': [gender],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        #  Perform one-hot encoding or mapping if the model was trained with encoded categories
        # (Adjust according to how you encoded 'Geography' and 'Gender' during training)
        input_data = pd.get_dummies(input_data, drop_first=True)

        # Align columns with model’s expected input
        # Ensure missing columns are added with 0s
        model_columns = scaler.feature_names_in_  # assumes the scaler was fitted with training data columns
        input_data = input_data.reindex(columns=model_columns, fill_value=0)

        #  Scale the input
        scaled_input = scaler.transform(input_data)

        #  Predict using the model
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0][1]

        prediction = int(prediction)
        prediction_proba = round(prediction_proba * 100, 2)

    return render_template('index.html', prediction=prediction, prediction_proba=prediction_proba)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
