import base64
import json
import cv2
import joblib
import numpy as np
from wavelet import w2d

__model = None

def get_cv2_image_from_bs4_string(bs4_string:str):

    if "data" in bs4_string:

        encoded_data = bs4_string.split(",")[1]

    else:
        encoded_data = bs4_string

    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)

    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    return img


def get_cropped_image_if_2_eyes(image_path:str, image_base64_data):

    face_cascade = cv2.CascadeClassifier("./opencv/haarcascades/haarcascade_frontalface_default.xml")

    eye_cascade = cv2.CascadeClassifier("./opencv/haarcascades/haarcascade_eye.xml")


    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_bs4_string(image_base64_data)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   
    faces = face_cascade.detectMultiScale(gray)
    

    cropped_faces = []
    
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        if len(eyes) >= 2:
            cropped_faces.append(roi_color)

    return cropped_faces







def classify_image(image_bs4_data, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_bs4_data)

    result = []

    for img in imgs:
        scalled_raw_image = cv2.resize(img, (32, 32))
        
        img_har = w2d(img, "db1", 5)
        scallled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_image.reshape(32*32*3,1), scallled_img_har.reshape(32*32,1)))

        lenght_of_img_array = (32 * 32 * 3) + (32 *32)

        final = combined_img.reshape(1, lenght_of_img_array).astype(float)

        probabilities = np.round( __model.predict_proba(final)*100, 2).tolist()[0]


        class_dictionary = {}

        for name , _class in __class_name_to_number.items():

            class_dictionary[name] = probabilities[_class]

        result.append({

            "class": class_number_to_name(__model.predict(final)[0]),

            "class_dictionary": class_dictionary

        })
            
    return result


def load_saved_artifacts():
    print("loading saved artifacts......")

    global __class_name_to_number

    global __class_number_to_name




    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)

        __class_number_to_name = {v:k for k, v in __class_name_to_number.items()} 

    global __model



    if __model is None:
        with open("./artifacts/saved_model.pkl", "rb") as f:
            __model = joblib.load(f)

    print("loading saved artfacts done....")




def get_bs4_test_image():
    with open("bs4.txt") as f:
        return f.read()





def class_number_to_name(class_name):
    return __class_number_to_name[class_name]





if __name__  == "__main__":

    load_saved_artifacts()

    # print(classify_image(get_bs4_test_image(), None))
    print(classify_image(None, "./test_images/Buhari-4-1.png"))
