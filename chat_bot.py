import csv
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import warnings
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, BooleanField
from wtforms.validators import DataRequired
import re


app = Flask(__name__)


warnings.filterwarnings("ignore", category=DeprecationWarning)


training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y
reduced_data = training.groupby(training['prognosis']).max()

# mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)


clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)

model = SVC()
model.fit(x_train, y_train)


importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols


name = ""
cnf_d = []
sym = ''
severarity = ''
symptoms_exp = []
symptoms_list = []
sentence = precution_list = None
number_of_days = None
disease = None
des = None
severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

feature_name = []
disease_input = ''
tree_ = None
present_disease = None
symptoms_dict = {}


@app.route('/', methods=['GET', 'POST'])
def get_name():
    global name
    if request.method == 'POST':
        name = str(request.form.get('name')).capitalize()
        return redirect(url_for('symptoms'))
    return render_template('form.html')


@app.route('/symptoms', methods=['GET', 'POST'])
def symptoms():
    global sym, cnf_d
    if request.method == 'POST':
        symptom = str(request.form.get('symptom'))
        cnf, cnf_dieses = tree_to_code(clf, cols, symptom)
        print("one", cnf, "two", cnf_dieses)
        while True:
            if cnf == 1:
                cnf_d = cnf_dieses
                sym = symptom
                return redirect('sevear')
            else:
                return render_template('name.html', name=name, error="Enter valid symptom.")
    return render_template('name.html', name=name)


@app.route('/sevear', methods=['GET', 'POST'])
def sevear():
    global severarity, cnf_d
    if request.method == 'POST':
        try:
            severarity = int(request.form.get('severarity'))
            return redirect(url_for('symlist'))
        except Exception as error:
            print(error)
            return render_template('sevear.html', name=name, symptom=sym, diesase=cnf_d, error="choose between numbers")
    return render_template('sevear.html', name=name, symptom=sym, diesase=cnf_d)


@app.route('/symlist', methods=['GET', 'POST'])
def symlist():
    global cnf_d, name, sym, severarity, symptoms_exp, symptoms_list
    if request.method == 'POST':
        symptoms_exp = request.form.getlist('symptoms_list')
        return redirect(url_for('number_day'))
    symptoms_list = recurse(0, 1)
    print("list are", symptoms_list)
    return render_template('list.html', name=name, symptom=sym, diesase=cnf_d, severarity=severarity, symptoms_list=symptoms_list)


@app.route('/day', methods=['GET', 'POST'])
def number_day():
    global symptoms_exp, symptoms_list, sentence, precution_list, number_of_days, des, disease
    if request.method == 'POST':
        day = int(request.form.get('day'))
        number_of_days = day
        sentence, precution_list, disease, des = give_result(day)
        return redirect(url_for('result'))

    return render_template('day.html', name=name, symptom=sym, diesase=cnf_d, severarity=severarity, symptoms_list=symptoms_list, symptoms_exp=symptoms_exp)


@app.route('/result', methods=['GET', 'POST'])
def result():
    global symptoms_exp, symptoms_list, sentence, precution_list, number_of_days, des, disease
    return render_template('result.html', name=name, symptom=sym, diesase=cnf_d, severarity=severarity, symptoms_list=symptoms_list, symptoms_exp=symptoms_exp, sentence=sentence, precution_list=precution_list, days=number_of_days, des=des, disease=disease)


# ------------------------------------------------------------------------------


def readn(nstr):
    engine = pyttsx3.init()

    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)

    engine.say(nstr)
    engine.runAndWait()
    engine.stop()


for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index


def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum = sum+severityDictionary[item]
    if ((sum*days)/(len(exp)+1) > 13):
        print("You should take the consultation from doctor. ")
        return "You should take the consultation from doctor. "

    else:
        print("It might not be that bad but you should take precautions.")
        return "It might not be that bad but you should take precautions."


def getDescription():
    global description_list
    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)


def getSeverityDict():
    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('MasterData/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)


def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if (len(pred_list) > 0):
        return 1, pred_list
    else:
        return 0, []


def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))


def tree_to_code(tree, feature_names, disease_in):
    global feature_name, disease_input, tree_
    disease_input = disease_in
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    conf, cnf_dis = check_pattern(chk_dis, disease_in)
    return conf, cnf_dis


def recurse(node, depth):
    global feature_name, disease_input, tree_, present_disease
    symptoms_present = []
    indent = "  " * depth
    if tree_.feature[node] != _tree.TREE_UNDEFINED:
        name = feature_name[node]
        threshold = tree_.threshold[node]

        if name == disease_input:
            val = 1
        else:
            val = 0
        if val <= threshold:
            val1 = recurse(tree_.children_left[node], depth + 1)
            return list(val1)
        else:
            symptoms_present.append(name)
            val2 = recurse(tree_.children_right[node], depth + 1)
            return list(val2)
    else:
        present_disease = print_disease(tree_.value[node])
        red_cols = reduced_data.columns
        symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero(
        )]
        return list(symptoms_given)


def give_result(num_days):
    global symptoms_exp, present_disease
    second_prediction = sec_predict(symptoms_exp)
    sentence = calc_condition(symptoms_exp, num_days)
    disease = None
    des = None
    if (present_disease[0] == second_prediction[0]):
        disease = present_disease[0]
        des = description_list[present_disease[0]]

    else:
        disease = second_prediction[0]
        des = description_list[second_prediction[0]]

    precution_list = precautionDictionary[present_disease[0]]
    return sentence, precution_list, disease, des


getSeverityDict()
getDescription()
getprecautionDict()

if __name__ == '__main__':
    app.run(debug=True)
