import cv2
import argparse
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

arr_out = []
arr_result = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
model=load_model('fmodelwts.h5')

def sortcnts(cnts):                 # to sort the contours left to right
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][0], reverse=False))
 
	return (cnts)

def test(a,b,c,d,imd):                # to predict the character present in the region of interest
    test=imd[b:b+d,a:a+c]
    _,test_image = cv2.threshold(test,100,255,cv2.THRESH_BINARY)
    test_image= cv2.copyMakeBorder(test_image,10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255))
    test_image = cv2.medianBlur(test_image.copy(),3)
    test_image = cv2.resize(test_image.copy(),(64,64),interpolation = cv2.INTER_AREA)
    t = test_image.copy()
    cv2.resize(test_image,(64,64))
    test_image=(image.img_to_array(test_image))/255
    test_image=np.expand_dims(test_image, axis = 0)
    result=model.predict(test_image)  
    np.reshape(result, 36)
    high = np.amax(test_image)
    low = np.amin(test_image)
    if high != low:
        #print(result)
        maxval = np.amax(result)
        index = np.where(result == maxval)
        #print('\n','Predicted Character:',arr_result[index[1][0]],'\n')
        cv2.imshow('grg', t)
        cv2.waitKey(0)
        #print(maxval, ' ', arr_result[index[1][0]])
        arr_out.append(arr_result[index[1][0]])


def pred(image_in):    
    im = image_in.copy()
    img = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        
    blur = cv2.bilateralFilter(img.copy(),9,75,75)
    kernel = np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]])
    img = cv2.filter2D(blur, -1, kernel)
    _, thresh = cv2.threshold(img.copy(), 100, 255, cv2.THRESH_BINARY)
    #median = cv2.medianBlur(thresh,5)
    
    
    cv2.imshow('thresh', thresh)
    cv2.waitKey(0)
    _,contours, h = cv2.findContours(thresh.copy(),cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #contours = sortcnts(contours)
    sum1 = 0
    maxar = 0
    
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        sum1 += (w*h)
        if cv2.contourArea(cnt)>maxar:
            maxar = cv2.contourArea(cnt)
    avg = sum1//len(contours)
    maxar = 10
    
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,255),2)
        test(x,y,w,h,img)
            
    final = ""
    i = 0
    for ch in arr_out:
        i += 1
        final = final+ch
      
    print('\n',final)

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
            help = 'path to input image')
ap.add_argument('-c', '--config', required=True,
            help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
            help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
            help = 'path to text file containing class names')
args = ap.parse_args()

image1 = cv2.imread(args.image)
Width = image1.shape[1]
Height = image1.shape[0]
scale = 0.00392

classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    net = cv2.dnn.readNet(args.weights, args.config)
    blob = cv2.dnn.blobFromImage(image1, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    print(label)

outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w / 2
            y = center_y - h / 2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

for i in indices:
    i = i[0]
    box = boxes[i]
    x = round(box[0]) if round(box[0])>=0 else 0
    y = round(box[1]) if round(box[1])>=0 else 0
    w = round(box[2]) if round(box[2])>=0 else 0
    h = round(box[3]) if round(box[3])>=0 else 0
    
    roi = image1[y:y+h,x:x+w]
    if str(classes[class_ids[i]]) == "car":
        cv2.imshow('roi',roi)
        cv2.waitKey()
        im = roi.copy()
        r,c,l = im.shape
        print(r,c)
        
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        #kernel = np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]])
        #gray = cv2.filter2D(gray, -1, kernel)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        #new = cv2.equalizeHist(gray)
        _,thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow('esrtf',thresh.copy())
        cv2.waitKey()
        edged = cv2.Canny(thresh, 170, 200)
        (_,cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#removed new
        #cnts=sorted(cnts, key=lambda x: cv2.contourArea(x), reverse = True)
        areal = r*c*0.019
        areau = r*c*0.025
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)            
            gg = im.copy()
            perimeter = 2*(w+h)
            if w*h > areal and w*h < areau:
                cv2.rectangle(gg, (x,y), (x+w, y+h), (255,0,255), 2)
                cv2.imshow('roi_number_plate',im[y:y+h,x:x+w])
                cv2.waitKey()
                pred(im[y:y+h,x:x+w])

cv2.destroyAllWindows()
