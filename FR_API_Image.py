## package imports
#from __future__ import print_function
#from flask import Flask, jsonify, request, make_response, send_file
#import FaceTraining as ft
#from keras import backend as K
#import test2 as t
#import json
#import excelsheet_entry as Ex
#
#app = Flask(__name__)
#
## set the path
#path_main = 'F:/'
#
#@app.route('/')
#def index():
#    return "Face Recognition !!"
#
#@app.errorhandler(404)
#def not_found(error):
#    return make_response(jsonify({'message': 'File Not found','code':204,'data':0}), 204)
#
## to predict the detected person    
#@app.route('/predict', methods = ['POST'])
#def predict():
#    K.clear_session()
#    person_dict = {}
#    
#    # receive the image of a person
#    f = request.files['TestImage']   
#    
#    # save the image in a folder named upvids
#    f.save(path_main + 'upvids/' + f.filename)
#    
#    # use the function - faces to predict the person
#    person_dict = t.faces(f.filename)
#    
#    # save the output dictionary in a json file - predict_person
#    with open('person_dict' + '.json','w') as outfile: 
#        json.dump(person_dict, outfile, sort_keys = True, indent = 4)
#    
#    # call the function excel_entry to make entries in excel sheet    
#    Ex.excel_entry(person_dict)
#            
#    return jsonify({'message':'Success!','data':person_dict,'code':200}),200
#
#
## train the model   
#@app.route('/train', methods = ['POST'])
#def train():
#    K.clear_session()
#    FaceEncodings, knownFaces = ft.FaceTraining.train()
#    ft.FaceTraining.storeData(FaceEncodings, knownFaces)
#    return jsonify({'message':'Success!','data':0,'code':200}),200
#
#
#if __name__ == '__main__':
#    app.run(host='0.0.0.0')
#


# package imports
from __future__ import print_function
from flask import Flask, jsonify, request, make_response
import FaceTraining as ft
from keras import backend as K
import Predict_API2 as t
import json
import excelsheet_entry as Ex
import os
app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

# set the path
path_main = '/home/iconflux/Desktop/python/RND/face_detection_frame/'

@app.route('/')
def index():
    return "Face Recognition !!"

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'message': 'File Not found','code':204,'data':0}), 204)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# to predict the detected person    
@app.route('/predict', methods = ['POST'])
def predict():
    K.clear_session()
    person_dict = {}
    
    if 'file' not in request.files:
        return jsonify({'messege':'Invalid Parameters!','code':201}),201

    # receive the image of a person
    f = request.files['file']

    if f.filename == '':
        return jsonify({'messege':'Invalid Parameters!','code':201}),201
    if allowed_file(f.filename) is False :
        return jsonify({'messege':'Invalid file!','code':202}),202   
    
    # save the image in a folder named upvids
    f.save(path_main + 'upvids/' + f.filename)
    
    # use the function - faces to predict the person
    person_dict = t.faces(f.filename, path_main)
    
    # save the output dictionary in a json file - predict_person
    with open('person_dict' + '.json','w') as outfile: 
        json.dump(person_dict, outfile, sort_keys = True, indent = 4)
    
    # call the function excel_entry to make entries in excel sheet    
    Ex.excel_entry(person_dict)
    os.remove(path_main + 'upvids/' + f.filename)        
    return jsonify({'message':'Success!','data':person_dict,'code':200}),200


# train the model   
@app.route('/train', methods = ['POST'])
def train():
    K.clear_session()
    FaceEncodings, knownFaces = ft.FaceTraining.train()
    ft.FaceTraining.storeData(FaceEncodings, knownFaces)
    return jsonify({'message':'Success!','data':0,'code':200}),200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003)


