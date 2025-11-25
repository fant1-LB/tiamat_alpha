from app.app import app, Config  # Importation de l'application Flask
from flask import render_template, request, flash, redirect, url_for, abort, send_file  # Fonctions Flask pour gérer les requêtes et réponses
from flask_wtf import FlaskForm
from flask_uploads import configure_uploads



import os
import sys
from pathlib import Path
import shutil
sys.path.append(str(Path(__file__).resolve().parent/"src"))


# scripts_path = root /'src'/ 'scripts'
# print(scripts_path)
# if str(scripts_path) not in sys.path:
#     sys.path.append(str(scripts_path))

# from scripts import *
from ..src.scripts.get_training_data import *
# data_preparation_and_training, statistics_for_training
from ..src.scripts.statistics_for_training import *
from ..src.scripts.data_preparation_and_training import *
from ..src.scripts.predicting_anc_checking_yolo_results import *
from ..src.models.formulaires import images as images_uploadees
from ..src.models.formulaires import *

@app.route("/",methods=['GET', 'POST'])
def accueil():
    form = NomDuProjet()
    

    return render_template("/pages/accueil.html" , form = form)
   

@app.route("/accueil_projet",methods=['GET', 'POST'])
def accueil_projet():
    form = NomDuProjet()
    if form.validate_on_submit():
            project_name= form.nom.data
            app.config['CURRENT_PROJECT_NAME']= project_name
         
    else:
        print(form.errors)
        
        
    project_name = app.config['CURRENT_PROJECT_NAME']
    print(project_name)
    if (Path.cwd() / project_name).is_dir():
        pass
    else:
        shutil.copytree(Path.cwd() / 'project', project_name)
    
    form2=ImportImages()
        
    return render_template("/pages/import_images.html", project_name=project_name, form2 = form2, form= form)

@app.route("/upload", methods = ['GET', 'POST'])
def upload():
    
    print(app.config['CURRENT_PROJECT_NAME'])
    project_name = app.config['CURRENT_PROJECT_NAME']
    app.config['UPLOADED_IMAGES_DEST'] = os.path.join(project_name, "image_inputs", "ground_truth_images")
    form = ImportImages()
    form2= ImportImages()
    configure_uploads(app, images_uploadees)
    root=os.getcwd()
    chemin_images = os.path.join(root,project_name, "image_inputs", "ground_truth_images")
    chemin_labels =os.path.join(root,project_name, "annotations", "ground_truth")
    if form.validate_on_submit():
        files2=[]
        for fichier in form.fichiers.data:
            images_uploadees.save(fichier)
            files2.append(fichier.filename)
        # Get the list of files from webpage
       
        return render_template ("/pages/upload_full.html", project_name= project_name, files2 = files2, form=form, form2=form2, chemin_images=str(chemin_images), chemin_labels=str(chemin_labels) )
    else:
        print(form.errors)
        print("erreur!!!")


@app.route("/label_lancement",methods=['GET', 'POST'])
def consignes() :

    root = os.getcwd()
    
    # print(root)
    project_name = app.config['CURRENT_PROJECT_NAME']

    clean_image_name(os.path.join(root, project_name))
    os.environ['LOCAL_FILES_DOCUMENT_ROOT'] = f'{root}'
    os.environ['LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED'] = 'true'

    # Launch Label Studio
    try:
        # print("Starting Label Studio... Don't forget to load the storage path to your images in your project settings !")
        # Bonne idée de maj à chaque lancement? Hit sunt dracones ou qqchose
        os.system('pip install -U label-studio')
        os.system('label-studio')
        return render_template("/pages/lancement_label_studio.html", project_name=project_name)
    except Exception as e:
        print(f"Error while starting Label Studio: {e}")
        return render_template("/pages/erreur.html", project_name=project_name)
    


@app.route("/dataset_statistics", methods=['GET', 'POST'])
def dataset_statistics():
    project_folder =os.path.join(os.getcwd(), app.config['CURRENT_PROJECT_NAME'])
    create_dataset(project_folder, manually_downloaded=False)
    print("dataset created")
    # Create the statistic folder
    create_stats_folder(project_folder)
    
    clean_LS(project_folder, annotated_with_LS=False)
    
    encoding(project_folder)
    
    annotations_per_img(project_folder)
    
    classes_distribution(project_folder)
    print("class distributon done")
    get_global_results(project_folder)
    print("got global results")
    
    return render_template("/pages/class_distrib.html",class_distibution_path=url_for('serve_dataset_image'), project_name=app.config['CURRENT_PROJECT_NAME'])
@app.route('/dataset_stats')
def serve_dataset_image():
    project_name=app.config['CURRENT_PROJECT_NAME']
    filename='class_distribution.png'
    file_path = Path.cwd() / "data" / project_name / "dataset_statistics" / filename
    if file_path.exists():
        return send_file(file_path, mimetype='image/png')
    return "File not found", 404


@app.route("/training_setup", methods=['GET', 'POST'])
def training_setup():
    return render_template("/pages/training_setup.html")
@app.route("/training", methods=['GET', 'POST'])
def training():
    project_folder = app.config['CURRENT_PROJECT_NAME']
    if request.method == 'POST':
        nombre_epoch = int(request.form.get("epochs"))
        model_file = request.files['model']
        if model_file and model_file.filename != "":
            print("tranquille oh ! j'ai pas eu le temps de faire ça encore!!!")
            
        else :
            model = "yolo11n.pt"
    clean_comma(project_folder)
    generate_transformed_data(project_folder)
    create_training_dataset(project_folder, model, preexisting_distribution=False)

#Config du réentraînement
    use_model = 'yolo11n.pt' # to be changed as needed, by default use 'yolov11x.pt'
    img_size = 640 # to be changed as needed, by default use 640
    epochs = nombre_epoch # to be changed as needed
    batch = -1 # to be changed as needed, by default use 8 or or -1 for AutoBatch
    workers = 8 # to be changed as needed, by default 24, or 8 (https://docs.ultralytics.com/modes/train/#train-settings)
    # label_smoothing = 0.1 # to be changed as needed,by default 0. Can improve generalization
    dropout = 0.1 # Elimine aléatoirement 10% connaissance à chaque époque

    yolo_training(project_folder, use_model, img_size, epochs, batch, workers, dropout, pretrained_model=None)

    model_path =dispatch_data(project_folder, use_model, img_size, 
                    epochs, batch, workers, dropout, 
                    pretrained_model=None, interrupted_model_folder=False)
    cwd = Path.cwd()
    model_path= Path(cwd / model_path)
    app.config['LAST_MODEL_PATH'] = model_path
    form2=ImportImages()
    return render_template("/pages/entrainement_fini.html", model_path = model_path, form2= form2)

@app.route("/test_modele", methods=['GET', 'POST'])
def test_upload():
    project_name = app.config['CURRENT_PROJECT_NAME']
    app.config['UPLOADED_IMAGES_DEST'] = os.path.join(project_name, "image_inputs", "eval_images")
    configure_uploads(app, images_uploadees)
    form2 = ImportImages()
    root = os.getcwd()
    abspathtoimages = os.path.join(root,app.config['UPLOADED_IMAGES_DEST'])
    if form2.validate_on_submit():
        files2=[]
        for fichier in form2.fichiers.data:
            images_uploadees.save(fichier)
            files2.append(fichier.filename)
    process_images_with_yolo(abspathtoimages, app.config['LAST_MODEL_PATH'])
    yolo_to_csv(abspathtoimages, app.config['LAST_MODEL_PATH'])
    get_ls_for_local_files(abspathtoimages, app.config['LAST_MODEL_PATH'])
    get_labeling_code(abspathtoimages, app.config['LAST_MODEL_PATH'])
    return render_template("/pages/bravo.html", current_folder = os.getcwd())

@app.route("/choisir_modele",methods=['GET', 'POST'])
def choisir_modele():
    form = CheminDuModele()
    project_folder = app.config['CURRENT_PROJECT_NAME']
    liste_modeles = get_model_list(project_folder)

    return render_template("/pages/choisir_modele.html" , form = form, liste_modeles =liste_modeles, project_folder=project_folder)

@app.route("/modele_choisi_main")
def modeles():
    project_name = app.config['CURRENT_PROJECT_NAME']
    
    form = CheminDuModele()
    if form.validate_on_submit():
            model_path= form.chemin.data
            app.config['LAST_MODEL_PATH']= model_path
    
    else:
       print(form.errors)
       return render_template("/pages/erreur.html")
    app.config['UPLOADED_IMAGES_DEST'] = os.path.join(project_name, "image_inputs", "eval_images")
    configure_uploads(app, images_uploadees)
    form2 = ImportImages()
    root = os.getcwd()
    abspathtoimages = os.path.join(root,app.config['UPLOADED_IMAGES_DEST'])
    
    process_images_with_yolo(abspathtoimages, app.config['LAST_MODEL_PATH'])
    yolo_to_csv(abspathtoimages, app.config['LAST_MODEL_PATH'])
    get_ls_for_local_files(abspathtoimages, app.config['LAST_MODEL_PATH'])
    get_labeling_code(abspathtoimages, app.config['LAST_MODEL_PATH'])
         
   
    return render_template("/pages/bravo.html", form=form, project_name=project_name, current_folder = os.getcwd())