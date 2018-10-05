# package imports
import face_recognition
import cv2
import numpy as np, os,time
import pickle
from string import digits

# initialisations
flag = 0
time_interval = 60

# define function for making the prediction
def faces(image, dir_path):
    
    # declaring a dictionary to store faces recognized
    person_dict = {}
    # path of the folder where the image to be processed is saved
    Face_path = dir_path + "upvids/"
    # path where Pickle file is stored
#    path = os.path.dirname(os.path.abspath("__file__")) + '/'    
    embeddingsPath = dir_path + 'Encoding_Data/'   
    
    # path where the training data set is stored
#    faceDbPath = path + 'FacesDB/'
    
    thresh =  0.1  # Threshold
    cnt=0
    
    # getting encoded data from the pickle file
    data = [d for d in os.listdir(embeddingsPath) if '.DS_Store' not in d] [0] 
    file ='FaceEncodingsModel.pkl'
    with open(embeddingsPath + file,'rb') as f:
        data = pickle.load(f)
    
    # assigning the model and different classes (no. of classes = no. of people in training data)
    model = data
    person = model.classes_
   
    try:
        # loading the image
        frame = face_recognition.load_image_file(Face_path + image)
        # determining the face encodings of the image       
        faceEncodings = face_recognition.face_encodings(frame)[0]
        # calculating probability of match with every other face in database
        prob = model.predict_proba(np.array(faceEncodings).reshape(1,-1))[0]
        # finding index with highest probability
        index = np.argmax(prob)
        
        # comparing the probability value with the threshold, if it is less than threshold - unknown face
        if np.max(prob) > float(thresh):
            # fetching name of the person
            remove_digits = str.maketrans('', '', digits)
            name = person[index].translate(remove_digits)
            # if name is not there in dictionary, it is added in the dictionary 
            if name not in person_dict:
                    person_dict[name] = time.time()
            # recognised images saved in the folder named RecognizedFaces        
            path = os.path.dirname(os.path.abspath("__file__")) + '/'  + 'RecognizedFaces' + '/' + str(person[index]) + '.jpg'
            cv2.imwrite(path, frame)
        else :
            # unkown images saved in the folder named - Unknown
            if not os.path.exists(dir_path + 'Unknown'):
                os.mkdir(dir_path + 'Unknown')
        
            path_for_Unknown = dir_path + 'Unknown' + '/' + str(cnt) + '.jpg'
            cv2.imwrite(path_for_Unknown, frame)
            cnt+=1
           
    except Exception as e:
        print('error')
        print(e)
    
#    print("Person_dict",person_dict)
    return person_dict


