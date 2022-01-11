import numpy as np
import cv2
import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
from keras.models import load_model
import threading
import os 
import signal
import sys
import time


emotion_model = load_model('emotion_model.h5')

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

emoji_dict={0:"./emojis/angry.png", 1:"./emojis/disgusted.png", 2:"./emojis/fearful.png", 3:"./emojis/happy.png", 4:"./emojis/neutral.png", 5:"./emojis/sad.png", 6:"./emojis/surpriced.png"}


# this is the predicted emotion for each captured frame
# lets make angry as defaut so that on first frame error will not come 
pridicted_emotion = 0 

# Time after which the camera will click a picture that
# will be predicted by the model. It is in milliseconds
camera_capturing_interval = 600


camera_captureing_interval_seconds = 0.4


def capture_live_video():
    """This function responsibility is to capture video frame and predict
    emotion of person update the cam window by making recangle around
    face and show the emotion in text above the rectangle.
    This function calls itself to capture the image after some period.
    """
    global pridicted_emotion
    global root
    global cam_window

    # this is the current frame we are predicting. 
    # we can consider it as the number of picture some camera took
    # when you go on a vacation. This number will tell how many
    # predictions were there on how many captured frames
    frame_number = 0

    # Live camera. 0 = live, video_path string = play video       
    camera = cv2.VideoCapture(0)                                 
    if not camera.isOpened():                             
        print("cant open the camera1")

    # length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))

    # print(length, frame_number)

    frame_number += 1
    # if frame_number >= length:
    #     exit()

    # 0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.
    ##USED## 1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
    # 2. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file
    # 3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
    # 4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
    # 5. CV_CAP_PROP_FPS Frame rate.
    # 6. CV_CAP_PROP_FOURCC 4-character code of codec.
    # 7. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
    # 8. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
    # 9. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
    # 10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
    # 11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).
    # 12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).
    # 13. CV_CAP_PROP_HUE Hue of the image (only for cameras).
    # 14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).
    # 15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).
    # 16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
    # 17. CV_CAP_PROP_WHITE_BALANCE Currently unsupported
    # 18. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
    
    # to set the camera frame setting accordingly. Here we need zero based index frame. ie frame values start from 0 rather than 1.
    camera.set(1, frame_number)

    is_frame_read_success, video_frame = camera.read()

    # Never resize. change the position of the frame. Else you video got laggy
    # video_frame = cv2.resize(video_frame, (600, 500))

    # A blue bounding box around face
    # to understand CascadeClassifier https://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0
    face_bounding_box = cv2.CascadeClassifier('C:/3-Programming/EmoDec/.emodec_venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    
    try:
        # converting colored frame to grayscale as training data was in grayscale
        gray_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        print('error in video frame cant get gray frame')
        return

    # Check paramaters of detectMultiScale here https://stackoverflow.com/questions/36218385/parameters-of-detectmultiscale-in-opencv-using-python
    num_faces = face_bounding_box.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Each face will give you the coordinate of point, height and width so that a rectangle can be formed
    for (x, y, w, h) in num_faces:

        # TODO: how rectangle is created in cv2 documentation
        cv2.rectangle(video_frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)

        # cropping image of face from full frame of video according to trainning data of model
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        
        # predicting cropped face image
        prediction = emotion_model.predict(cropped_img)
        
        # give index of maximum probabiliy of emotion 
        maxindex = int(np.argmax(prediction))

        # print(emotion_dict[maxindex])

        # Text of emotion above the face box
        cv2.putText(video_frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        pridicted_emotion = maxindex
        print('predicted image', emoji_dict[pridicted_emotion])
        
    if is_frame_read_success is None:
        print ("Major error!")
    elif is_frame_read_success:
        # here we are creating a copy of captured video frame for predicting emotion.
        captured_frame = video_frame.copy()

        # converting captured frame to RGB(red, green, blue) real color
        # BGR (blue, green, red)
        pic = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)

        # converting array of pixels into actual image
        img = Image.fromarray(pic)

        # now converting that image object in binary into a proper image object used PIL
        imgtk = ImageTk.PhotoImage(image=img)

        # cam_window is used to insert the captured image by video camera then insert that captured
        # frame to cam_window frame. which looks like camera is recording the reaction of the person
        cam_window.imgtk = imgtk

        # her econfigure is needed to updated the image at that lable or the window that is created
        cam_window.configure(image=imgtk)

        # # then update the root as we need to refresh root window to see the changes in cam window.
        # root.update()

        # then call the itself like a recursive function so that next frame should be caputured after 
        # required milliseconds and whole process of capturing frame and predicting emotion
        # can repeat itself.
        # cam_window.after(camera_capturing_interval, capture_live_video)

        show_avatar()


def show_avatar():
    """This function is used to show avatar with emotion in the UI on the right.
    Main intention to show avatar is to show the replica of human emotion.
    """
    global root
    global emoji_avatar
    global emoji_label

    # this is the avatar image of the emotion read by cv2
    avatar_image = cv2.imread(emoji_dict[pridicted_emotion])

    # here we changing the avatar image from BGR to RGB
    pic = cv2.cvtColor(avatar_image ,cv2.COLOR_BGR2RGB)

    # Then converting this array of pixels to binary image object
    image = Image.fromarray(pic)
    
    # now creating image object which can be manupulated. this 
    # image object is created through PIL library
    imgtk=ImageTk.PhotoImage(image=image)

    # now configuring emoji avatar window with the image object which is to be
    # show as human replica of emotion
    emoji_avatar.imgtk=imgtk
    
    # first the text is confugeres shown above the avatar
    emoji_label.configure(text=emotion_dict[pridicted_emotion],font=('arial',45,'bold'))
    
    # then the emoji avatar is configured to show the image of the avatar on emoji avatar
    # window
    emoji_avatar.configure(image=imgtk)

    # updating root window is important else you will not see the updated avatar in
    # in avatar window
    # root.update()

    # # show emoji avatar just when the camera image is captured and predicted.
    # emoji_avatar.after(camera_capturing_interval, show_avatar)


def signal_handler(signum, frame):
    global terminate

    terminate = True
    root.destroy()
    sys.exit()

def terminate_script():
    global terminate

    terminate = True
    root.destroy()
    sys.exit()

# if __name__ == '__main__':
    
terminate = False
signal.signal(signal.SIGINT, signal_handler)

# root is the whole window that tkinter opened as ui
root = tk.Tk()   

# Window at which the camera will record your videos
cam_window = tk.Label(master=root, padx=50, bd=10)
cam_window.pack(side=LEFT)
cam_window.place(x=50,y=50)

# avatar of emoji
emoji_avatar = tk.Label(master=root, bd=10)
emoji_avatar.pack(side=RIGHT)
emoji_avatar.place(x=900,y=150)

# Emoji avatar label window
emoji_label = tk.Label(master=root, bd=10, fg="#CDCDCD", bg='black')
emoji_label.pack()
emoji_label.place(x=960, y=50)

root.title("EmoDec")            
root.geometry("1400x700+100+10") 
root['bg']='grey'

# Quit button
exitbutton = Button(root, text='Quit', fg="red", command=terminate_script, font=('arial', 25, 'bold')).pack(side = BOTTOM)

while not terminate:
    print('Starting')
    capture_live_video()
    print('after capture live video')
    root.update_idletasks()
    root.update()
    print('after update')
    time.sleep(camera_captureing_interval_seconds)
    print('finish loop')
