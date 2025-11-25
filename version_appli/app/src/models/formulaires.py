from flask_wtf import FlaskForm
from wtforms import StringField, SelectField, SelectMultipleField, TextAreaField, SubmitField, FloatField, FileField
from wtforms.validators import DataRequired, Length
from flask_wtf.file import MultipleFileField, FileAllowed, FileRequired
from flask_uploads import UploadSet, IMAGES

class NomDuProjet(FlaskForm):
    nom=StringField("Nom du projet", validators=[DataRequired(),Length(min=0)],)
    # submit = SubmitField('Submit')

images = UploadSet('images', IMAGES)
class ImportImages(FlaskForm):
    fichiers=MultipleFileField("Fichiers", validators =[FileRequired(), FileAllowed(images, "Images svp")])
    # submit = SubmitField('Submit')

class CheminDuModele(FlaskForm):
    chemin=StringField("Chemin du modele", validators=[DataRequired(),Length(min=0)],)
    # submit = SubmitField('Submit')