from flask import Flask, render_template, request
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Flask app with correct template folder
app = Flask(__name__, template_folder="template")

# Load model and encoders with error handling
try:
    with open('loan_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('label_encoders.pkl', 'rb') as le_file:
        label_encoders = pickle.load(le_file)
except FileNotFoundError as e:
    print(f"Error: Could not load file - {e}")
    model = None
    label_encoders = None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predictor')
def predictor():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or label_encoders is None:
        return "Error: Model or encoders not loaded properly.", 500

    try:
        data = request.form
        features = [
            label_encoders['Gender'].transform([data['Gender']])[0],
            label_encoders['Married'].transform([data['Married']])[0],
            3 if data['Dependents'] == '3+' else int(data['Dependents']),
            label_encoders['Education'].transform([data['Education']])[0],
            label_encoders['Self_Employed'].transform([data['Self_Employed']])[0],
            float(data['ApplicantIncome']),
            float(data['CoapplicantIncome']),
            float(data['LoanAmount']),
            float(data['Loan_Amount_Term']),
            float(data['Credit_History']),
            label_encoders['Property_Area'].transform([data['Property_Area']])[0]
        ]
        
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]
        result = "Approved" if prediction == 1 else "Rejected"
        result_class = "approved" if prediction == 1 else "rejected"
        
        return render_template('result.html', result=result, result_class=result_class)
    
    except KeyError as e:
        return f"Error: Missing or invalid form field - {e}", 400
    except ValueError as e:
        return f"Error: Invalid input format - {e}", 400
    except Exception as e:
        return f"Error: Something went wrong - {e}", 500

if __name__ == '__main__':
    app.run(debug=True)
    