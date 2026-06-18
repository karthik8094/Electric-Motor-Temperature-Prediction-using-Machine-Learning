from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():

    values = [float(x) for x in request.form.values()]

    final = np.array(values).reshape(1, -1)


    prediction = model.predict(final)

    return render_template(
        'home.html',
        prediction_text=f'Predicted Temperature: {prediction[0]}'
    )

if __name__ == "__main__":
    app.run(debug=True)  