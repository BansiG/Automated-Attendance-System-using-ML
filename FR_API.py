## package imports
#from __future__ import print_function
#from flask import Flask, jsonify, make_response
#import FaceTraining as ft
#from keras import backend as K
#import Predict as t
#
## set the path
#path_main = 'F:/'
#
#app = Flask(__name__)
#
#@app.route('/')
#def index():
#    return "Face Recognition !!"
#
#@app.errorhandler(404)
#def not_found(error):
#    return make_response(jsonify({'message': 'File Not found','code':204,'data':0}), 204)
#
## predict method
## no inputs. Just save all the images in a folder names Faces    
#@app.route('/Predict', methods = ['GET'])
#def predict():
#    K.clear_session()
#    print("in the predict")
#    person_array = t.faces() # detected person names stored in person_array. faces function - predicts the person
#    
#    return jsonify({'message':'Success!','data':person_array,'code':200}),200
# 
## Train method 
## no inputs, just save all the faces in a folder named FacesDb    
#@app.route('/train', methods = ['POST'])
#def train():
#    K.clear_session()
#    
#    # getting faces encodings with the help of train function
#    FaceEncodings, knownFaces = ft.FaceTraining.train()
#    # storing the encodings in a pickle file for quick operation
#    ft.FaceTraining.storeData(FaceEncodings, knownFaces)
#    return jsonify({'message':'Success!','data':0,'code':200}),200
#
#
#if __name__ == '__main__':
#    app.run(host='0.0.0.0')
#

# package imports
from __future__ import print_function
from flask import Flask, jsonify, make_response, request
import FaceTraining as ft
from keras import backend as K
import Predict as t

# set the path
path_main = 'F:/'

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])


app = Flask(__name__)

@app.route('/')
def index():
    return "Face Recognition !!"

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'message': 'File Not found','code':204,'data':0}), 204)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# predict method
# no inputs. Just save all the images in a folder names Faces    
@app.route('/Predict', methods = ['GET'])
def predict():
    K.clear_session()
    
    if 'file' not in request.files:
        return jsonify({'messege':'Invalid Parameters!','code':201}),201

    # receive the image of a person
    f = request.files['file']

    if f.filename == '':
        return jsonify({'messege':'Invalid Parameters!','code':201}),201
    if allowed_file(f.filename) is False :
        return jsonify({'messege':'Invalid file!','code':202}),202 
    
    
    person_array = t.faces(path_main) # detected person names stored in person_array. faces function - predicts the person
    
    return jsonify({'message':'Success!','data':person_array,'code':200}),200
 
# Train method 
# no inputs, just save all the faces in a folder named FacesDb    
@app.route('/train', methods = ['POST'])
def train():
    K.clear_session()
    
    # getting faces encodings with the help of train function
    FaceEncodings, knownFaces = ft.FaceTraining.train()
    # storing the encodings in a pickle file for quick operation
    ft.FaceTraining.storeData(FaceEncodings, knownFaces)
    return jsonify({'message':'Success!','data':0,'code':200}),200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8003)



