from sklearn.cluster import KMeans
import random as rng
import cv2
import imutils
import argparse
from imutils import contours
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import *
from flask import *

app=Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def hello_world():
   return render_template('index.html')

# ImgPath = 'static/barefeet1.jpeg'
@app.route('/sizechart', methods = ['GET', 'POST'])
def sizechart():
    return render_template('sizechart.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict_size():
        print("hello")
        if request.method == 'POST':

            f = request.files['file']
            oimg = imread(f)

            # if not os.path.exists('output'):
            #     os.makedirs('output')
            preprocessedOimg = preprocess(oimg)
            cv2.imwrite('output/preprocessedOimg.jpg', preprocessedOimg)

            clusteredImg = kMeans_cluster(preprocessedOimg)
            cv2.imwrite('output/clusteredImg.jpg', clusteredImg)

            edgedImg = edgeDetection(clusteredImg)
            cv2.imwrite('output/edgedImg.jpg', edgedImg)

            boundRect, contours, contours_poly, img = getBoundingBox(edgedImg)
            pdraw = drawCnt(boundRect[1], contours, contours_poly, img)
            cv2.imwrite('output/pdraw.jpg', pdraw)


            croppedImg, pcropedImg = cropOrig(boundRect[1], clusteredImg)
            cv2.imwrite('output/croppedImg.jpg', croppedImg)
            

            newImg = overlayImage(croppedImg, pcropedImg)
            cv2.imwrite('output/newImg.jpg', newImg)

            fedged = edgeDetection(newImg)
            fboundRect, fcnt, fcntpoly, fimg = getBoundingBox(fedged)
            fdraw = drawCnt(fboundRect[2], fcnt, fcntpoly, fimg)
            cv2.imwrite('output/fdraw.jpg', fdraw)

            #print("feet size:",round(calcFeetSize(pcropedImg, fboundRect)*0.393701/10 , 2), "inches")

            Foot_Size = round(calcFeetSize(pcropedImg, fboundRect)*0.393701/10,2)
            #Foot_Size = (calcFeetSize(pcropedImg, fboundRect)/10)

            #Foot_Size= calcFeetSize(pcropedImg, fboundRect)/10
            #print(Foot_Size, "inches")

            return render_template('index.html', Foot_Size=Foot_Size)

if __name__ == '__main__':
    app.run(debug=True)