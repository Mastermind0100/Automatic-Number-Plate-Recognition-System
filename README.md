[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com) [![forthebadge](https://forthebadge.com/images/badges/kinda-sfw.svg)](https://forthebadge.com)<br>
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![LinkedIn-profile](https://img.shields.io/badge/LinkedIn-Atharva-blue.svg)](https://www.linkedin.com/in/atharva-hudlikar/)
# Automatic Number Plate Recognition (ANPR) System
## Synopsis
As the name suggests, this code guesses an area of region that might be the number plate of a car. The accuarcy of the code relies a bit on the lighting in the input image. <br/>
The detection is two fold. First, the code detects the region of image in which the car is present. This bit is achieved using pretrained network and code based on [YOLOv3](https://pjreddie.com/darknet/yolo/) network. A lot of my code also draws inspiration from [this amazing blog](https://medium.com/@ravidesetty/number-plate-detection-on-indian-car-vehicles-using-yolov2-8c99e1a259f5). <br/>
Then, the code attempts to guess the region of image where the number plate may be present. I have expanded on the idea I read about and explored in [this repository](https://github.com/shanmathik/Indian-License-Plate-Recognition-). <br/><br/>
P.S. The pretrained weights for YOLOv3 is too big to upload, so I am provideing the relevant link [here](https://pjreddie.com/media/files/yolov3-openimages.weights)

## How to Run the code?
**Step 1**: Clone the Repository:
```bash
$ git clone https://github.com/Mastermind0100/Automatic-Number-Plate-Recognition-System.git
```
Save the image you want to use for detection in the same directory as testimage.jpg (or whatever extention you are comfortable with).
<br/>
**Step 2**: Open cmd in the same directory (or open cmd and go to the directory where all the files are saved)<br/>
**Step 3**: Type the following command:
```bash
python new_yolo.py --image testimage.jpg --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
```
Note, the testimage.jpg is to be replaced by the image you want to test<br/>
**Step 4**: Hope you get the output you want! <br/><br/> 
P.S. Press any key to obtain further images :)<br/><br/>

## Input
![test12](https://user-images.githubusercontent.com/36445600/60386078-73d77d80-9aae-11e9-886f-2b8c869f17e4.jpg)

## Output
### Region of Image of Car
![roi1](https://user-images.githubusercontent.com/36445600/60386131-e8aab780-9aae-11e9-86c2-f6cfb0aecd1e.JPG)
### Region of Image of Number Plate
![nplate1](https://user-images.githubusercontent.com/36445600/60386232-1e9c6b80-9ab0-11e9-9fd2-b192c30e1cdb.JPG)
### Thresholded ROI of Number Plate
![nplate2](https://user-images.githubusercontent.com/36445600/60386245-3ffd5780-9ab0-11e9-82df-5206ea917577.JPG)

## Pro Tip
You can use this code in conjunction with my OCR (Optical Character Recognition) [repository](https://github.com/Mastermind0100/basic_ocr) to obtain a full fledged ANPR (Automatic Number Plate Recognition) system! :D
