# coding=utf-8

# Import libraries
from gen_html import fit_model
from flask import Flask, render_template, request
from wtforms import Form, TextField, validators, SubmitField, DecimalField, IntegerField, SelectField

# Create app
app = Flask(__name__)


# Define a form
class ReusableForm(Form):

    ParCh = IntegerField('Number of popular words to show:',
                         default=5,
                         validators=[validators.InputRequired(),
                                     validators.NumberRange(min=1, max=50,
                                                            message='Number must be between 1 and 50')])
    submit = SubmitField('Predict')


# Home page
@app.route('/', methods=['GET', 'POST'])
def home():
    # Create form
    form = ReusableForm(request.form)

    # On form entry and all entries validated
    if request.method == 'POST' and form.validate():
        # Extract information
        ParCh = int(request.form['ParCh'])
        # Send information to template
        return render_template('predict.html',
                               input=fit_model(ParCh=ParCh))

    return render_template('index.html',
                           form=form)


if __name__ == '__main__':
    print((' * Loading model and Flask starting server...\n * Please wait until server has fully started'))
    # Run app
    app.run(host='0.0.0.0', port=8888)
