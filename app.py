


import pickle


from flask import Flask,request, url_for, redirect, render_template
# import _pickle as pickle
import numpy as np


app = Flask(__name__)


model = pickle.load(open('naiv2_model.sav', 'rb'))

print(model)


@app.route('/')
def home():
    return render_template("forest.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    
    prediction=  model.predict_proba(final)
    print(prediction)
    #output='{0:.{1}f}'.format(prediction[0][1], 2)
    output = str(0.2)
    if output>str(0.5):
        return render_template('forest.html',pred='Your Forest is in Danger.\nProbability of fire occuring is {}'.format(output),bhai="kuch karna hain iska ab?")
    else:
        return render_template('forest.html',pred='Your Forest is safe.\n Probability of fire occuring is {}'.format(output),bhai="Your Forest is Safe for now")


if __name__ == '__main__':
    app.run()