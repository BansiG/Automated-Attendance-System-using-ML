
import numpy as np, os
import cv2

#import openface
import face_recognition
import pickle
from sklearn.svm import SVC

knownFaces = []
knownEncodings =[]
path = os.path.dirname(os.path.abspath("__file__")) + '/'
imageDir = path + 'FacesDB/'

count=0
folders = os.listdir(imageDir)
folders = [item for item in folders if item != '.DS_Store']
folders = [item for item in folders if item != 'Unknown']
svm = SVC(C=100.0,kernel='linear',gamma=0.001,probability=True,verbose=2)
        
class FaceTraining():
    
    def storeData( knownEncodings, knownFaces):
        
        print("Training Started...")
        
        svm.fit(knownEncodings,knownFaces)
        # =============================================================================
        # Storing the Encodings Data
        # =============================================================================
        with open(path+ 'Encoding_Data/' + 'FaceEncodings.pkl','wb') as f:
                        pickle.dump(knownFaces, f,  protocol=2)
        # =============================================================================
        # Storing the model.
        # =============================================================================
        with open(path+ 'Encoding_Data/' + 'FaceEncodingsModel.pkl','wb') as f:
                        pickle.dump(svm, f,  protocol=2)
        
        
        print("Training Finished...! Model can be found at {} location ".format(path+ 'Encoding_Data/'))
    

    # =============================================================================
    # Function to train on Data
    # =============================================================================
    def train():
        for folder in folders:
            #knownFaces.append(folder)
            images = [item for item in os.listdir(imageDir + folder +'/') if item != '.DS_Store']
            
            print('Starting {} which has {} Images'.format(folder,len(images)))
            
            for img in images:   
                image = cv2.imread(imageDir + folder +'/' + img)
                
                try:    
                #getting the face location
                    faceLcoation = face_recognition.face_locations(image)
                    #now getting faceEncodings
                    faceEncodings = face_recognition.face_encodings(image,faceLcoation)[0]    
                    #knownFaces[str(folder)] = faceEncodings
                except:
                    os.remove(imageDir + folder +'/' + img)
                    #print('count:' , count)
                    
                    continue
                knownFaces.append(folder)
                knownEncodings.append(faceEncodings)
            
            print('done with {} and number of not found Faces: total {} facesData '.format(folder, len(images)))
#            count = 0
        
        return knownEncodings, knownFaces





    