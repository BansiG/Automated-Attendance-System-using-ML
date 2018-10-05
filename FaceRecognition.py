import face_recognition
import cv2
import dlib
import numpy as np, os,time
import pickle
#import youtube_dl
import imutils
from imutils.video import FileVideoStream
from imutils.video import FPS
#from FaceTraining import FaceTraining
from string import digits

class FaceRecognition():
    
        
#    @staticmethod
#    def loadPickleFile(path, file='FaceEncodingsModel.pkl'):
#        data = [d for d in os.listdir(path) if '.DS_Store' not in d] [0] 
#        
#        with open(path + file,'rb') as f:
#            data = pickle.load(f)
#        
#        return data 
    def findFaces():
        Face_path = "F:/Faces"
        path = os.path.dirname(os.path.abspath("__file__")) + '/'    
        embeddingsPath = path+ 'Encoding_Data/'   
#        faceModel_path = 'shape_predictor_68_face_landmarks.dat'
        faceDbPath = path + 'FacesDB/'
        
        thresh =  0.6
        cnt=0
        person_array = []
        
        data = [d for d in os.listdir(embeddingsPath) if '.DS_Store' not in d] [0] 
        file ='FaceEncodingsModel.pkl'
        with open(embeddingsPath + file,'rb') as f:
            data = pickle.load(f)
        
        model = data
        person = model.classes_
        
#        landMarker = dlib.shape_predictor(faceModel_path) 
        
        for folder in os.listdir(Face_path):
            for frame in os.listdir(Face_path + '/' + folder):
                try:
                    faceLocs = face_recognition.face_locations(frame)
                    
                    faceEncodings = face_recognition.face_encodings(frame,faceLocs)
#                    faces = faceDetector(frame,1)
#                    len_faces = len(faces)
#                    print('len of Faces: ', len(faces))
                except Exception as e:
                    print('error')
                    print(e)
                    continue
                
                prob = model.predict_proba(np.array(faceEncodings).reshape(1,-1))[0]
                index = np.argmax(prob)
            
                if np.max(prob) > float(thresh):
                    text = str(person[index] + str(np.round_(np.max(prob),2)))

                    remove_digits = str.maketrans('', '', digits)
                    name = person[index].translate(remove_digits)
                    
                    if name not in person_array:
                        person_array.append(name)
                    
                    path = os.path.dirname(os.path.abspath("__file__")) + '/'  + 'RecognizedFaces' + '/' + str(text) + '.jpg'
                    cv2.imwrite(path, frame)
                else :
                    text = 'Unknown'
                    if not os.path.exists(faceDbPath + 'Unknown'):
                        os.mkdir(faceDbPath + 'Unknown')
                
                    path = faceDbPath + 'Unknown' + '/' + str(cnt) + '.jpg'
                    cv2.imwrite(path, frame)
                    cnt+=1
        print(person_array)

    def recognize(video):
    
        path = os.path.dirname(os.path.abspath("__file__")) + '/'    
        embeddingsPath = path+ 'Encoding_Data/'   
        faceModel_path = 'shape_predictor_68_face_landmarks.dat'
        faceDbPath = path + 'FacesDB/'
        
#        model = cls.loadPickleFile(embeddingsPath,file='FaceEncodingsModel.pkl')
        #getting the people list
        data = [d for d in os.listdir(embeddingsPath) if '.DS_Store' not in d] [0] 
        file ='FaceEncodingsModel.pkl'
        with open(embeddingsPath + file,'rb') as f:
            data = pickle.load(f)
        
        model = data
        person = model.classes_

#gettign the Frontal face area from the imag
        j = 0 
        faceDetector = dlib.get_frontal_face_detector()
        #getting landmark Points
        landMarker = dlib.shape_predictor(faceModel_path) 
        person_array = []    
        count=0
        cnt = 0
        len_faces = 0
        #cap = cv2.VideoCapture("vid2.mp4")
        
        cap = cv2.VideoCapture(video)
        fps = FPS().start()
  
        print("Input video is laoded... ")
        
        fourcc = cv2.VideoWriter_fourcc('X','2','6','4')
        writer = cv2.VideoWriter('TestVideo.avi',fourcc, 12.0, (int(450),int(400)))  
        writer = None 
        print("Entering the loop...")
        l = True
        
        while(l):
            ret, frame = cap.read()
            if(ret == True):
                frame =  imutils.rotate_bound(frame, 90)
                frame = imutils.resize(frame, width=450) 
                if(len_faces == 0):
                    j = 6
                else:
                    j = 1
                
                if(cnt%j ==0):
            
            
            #frame = imutils.rotate_bound(frame, 180)
            
             
                    (h, w) = frame.shape[:2]
                    #ret,frame = cap.read()
                    #print(ret)
                    if writer is None:
                        (h,w) = frame.shape[:2]
                        writer = cv2.VideoWriter('TestVideo.mp4', fourcc, 10.0, (w, h), True)
            
            # =============================================================================
                    print('Starting up')
                
                    try:
                        faceLocs = face_recognition.face_locations(frame)
                        
                        faceEncodings = face_recognition.face_encodings(frame,faceLocs)
                        faces = faceDetector(frame,1)
                        len_faces = len(faces)
                        print('len of Faces: ', len(faces))
                    except Exception as e:
                        print('error')
                        print(e)
                        continue
                    
                    thresh = 0.6
                
                
            
                    cv2.putText(frame, 'Threshold = {}'.format(str(thresh)), (0,50),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,0,0),2)
         
                    
                    for i in range (0,len(faces)):
                        newRect = dlib.rectangle(int(faces[i].left() ),
                                                int(faces[i].top() ),
                                                int(faces[i].right() ),
                                                int(faces[i].bottom() ))
                      
                        #Getting x,y,w,h coordinate
                        x = newRect.left()
                        y = newRect.top()
                        w = newRect.right() - x
                        h = newRect.bottom() - y
                        X,Y,W,H = x,y,w,h
                        
                        
                        prob = model.predict_proba(np.array(faceEncodings[i]).reshape(1,-1))[0]
                        index = np.argmax(prob)
                    
                        if np.max(prob) > float(thresh):
                            text = str(person[index] + str(np.round_(np.max(prob),2)))
                            cv2.rectangle(frame,(x, y), (x+w, y+h), (0,255,0),2)
                            cv2.putText(frame, text, (x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
                            
                            remove_digits = str.maketrans('', '', digits)
                            name = person[index].translate(remove_digits)
                            
                            if name not in person_array:
                                person_array.append(name)
                            
                            path = os.path.dirname(os.path.abspath("__file__")) + '/'  + 'RecognizedFaces' + '/' + str(text) + '.jpg'
                            cv2.imwrite(path, frame[y:y+h, x:x+w])
                        else :
                            text = 'Unknown'
                            cv2.rectangle(frame,(x, y), (x+w, y+h), (0,0,255),1)
                            cv2.putText(frame, text, (x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
          
                            if not os.path.exists(faceDbPath + 'Unknown'):
                                os.mkdir(faceDbPath + 'Unknown')
                        
                            path = faceDbPath + 'Unknown' + '/' + str(cnt) + '.jpg'
                            cv2.imwrite(path, frame[y:y+h,x:x+w])
         
                    cv2.imshow('frame',frame)
                    writer.write(frame)
            else:
                l = False
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break    
            cnt += 1
        print("Done processing the video ...")
        #cap.release()
        cv2.destroyAllWindows()

        writer.release()
        return person_array
    

fr = FaceRecognition()
video = "crazy9.mp4"
fr.findFaces()


