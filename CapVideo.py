#https://answers.opencv.org/question/200861/drawing-a-rectangle-around-a-color-as-shown/
#https://stackoverflow.com/questions/10948589/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-withcv
#https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
#https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces


import numpy as np
import cv2 
import argparse


cap = cv2.VideoCapture(0)

#Define the code and create VideoWriter object
    # fourcc = cv2.videoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi',fourcc,20.0,(*640,480))

# Setting boundaries for colors, this can be done as a matrix of multiple color boundaries
boundaries = [
   # ([100, 50, 30], [200, 100, 60])  # This detects a light blue... must have the order wrong
     ([0, 30, 150], [100, 140, 255])  # This recognized organge... specifically my orange BIC lighter
]


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Flips image horizontally, i.e. mirror image
    if ret == True:
        frame = cv2.flip(frame,1) # By using 1 we flip horizontal '0' would flip vertical

    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        #find the colors within the specified boundaries and apply the mask
        mask = cv2.inRange(frame, lower, upper)
        output = cv2.bitwise_and(frame, frame, mask = mask)

##############################################################
# Attatches a rectangle around the color mask set to it (currently orange)
    orangecnts = cv2.findContours(mask.copy(), 
                                cv2.RETR_EXTERNAL, 
                                cv2.CHAIN_APPROX_SIMPLE) [-2]
    if len(orangecnts) > 0:
        ornage_area = max(orangecnts, key = cv2.contourArea)
        (xg,yg,wg,hg) = cv2.boundingRect(ornage_area)
        cv2.rectangle(output, (xg,yg), (xg+wg, yg+hg), (0,255,0),2)

###############################################################

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    # cv2.imshow('frame', frame)

    cv2.imshow('frame', np.hstack([frame, output]))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()