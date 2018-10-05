
# package imports
import face_recognition
import cv2
import numpy as np, os
import pickle
from string import digits

# function for predicting person name from the face image stored in a folder named Faces
def faces(path):
    
    # path of the folder where all images are stored
    Face_path = path + "Faces"
    
    # path of the pickle file
    path = os.path.dirname(os.path.abspath("__file__")) + '/'    
    embeddingsPath = path+ 'Encoding_Data/'   
#    faceDbPath = path + 'FacesDB/'
    
    # threshold value
    thresh =  0.1
    cnt=0
    
    # declaring an array to store names of all detected faces
    person_array = []
    
    # loading the pickle file
    data = [d for d in os.listdir(embeddingsPath) if '.DS_Store' not in d] [0] 
    file ='FaceEncodingsModel.pkl'
    with open(embeddingsPath + file,'rb') as f:
        data = pickle.load(f)
    
    # assigning model and the classes. (no. of classes = no. of trained faces)
    model = data
    person = model.classes_
    
    # fetching images one by one from different folders
    for folder in os.listdir(Face_path):
        for image in os.listdir(Face_path + '/' + folder):
           
#            print(Face_path + '/' + folder + '/' + image)
   
            try:
                # loading the image
                frame = face_recognition.load_image_file(Face_path + '/' + folder + '/' + image)
                
                # determining the face encodings of the image
                faceEncodings = face_recognition.face_encodings(frame)[0]
   
#                print(faceEncodings.shape)
    
            except Exception as e:
                print('error')
                print(e)
                continue
            
            # calculating the probability of match with every other face and finding max probability
            prob = model.predict_proba(np.array(faceEncodings).reshape(1,-1))[0]
            # finidng index with max probability
            index = np.argmax(prob)
        
            # comparing the probability with the threshold
            if np.max(prob) > float(thresh):
#                text = str(person[index] + str(np.round_(np.max(prob),2)))
    
                remove_digits = str.maketrans('', '', digits)
                name = person[index].translate(remove_digits)
                
                # if name is not in array, then adding it in the array
                if name not in person_array:
                    person_array.append(name)
                
                # Saving the recognized faces in a folder - RecognizedFaces
                path = os.path.dirname(os.path.abspath("__file__")) + '/'  + 'RecognizedFaces' + '/' + str(person[index]) + '.jpg'
                cv2.imwrite(path, frame)
            else :
#               # saving the unkown faces in a folder - Unkown
                if not os.path.exists(path + 'Unknown'):
                    os.mkdir(path + 'Unknown')
            
                path_for_Unknown = path + 'Unknown' + '/' + str(cnt) + '.jpg'
                cv2.imwrite(path_for_Unknown, frame)
                cnt+=1
                
    print(person_array)
    return person_array