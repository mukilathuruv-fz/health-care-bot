from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import StringField, BooleanField
from wtforms.validators import DataRequired
app = Flask(__name__)


# forms

class LoginForm(FlaskForm):
    user = StringField(label='Enter your name', validators=[DataRequired()])


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
