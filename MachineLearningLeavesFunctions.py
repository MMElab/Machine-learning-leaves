# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 19:57:48 2023

@author: vinkjo
"""

## Functions
# Finds the closest and furthest point based on a skeleton
def filter_contours_by_size(contours, min_contour_area=100):
    filtered_contours = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area >= min_contour_area:
            filtered_contours.append(contour)
    return filtered_contours

def float_to_rgb(float_array):
    # Scale the float array to the 0-255 range
    scaled_array = (float_array * 255).astype(np.uint8)

    # Create an RGB format array with all channels having the same values
    rgb_array = np.stack((scaled_array, scaled_array, scaled_array), axis=-1)

    return rgb_array

def find_necrosis_pixels(image_with_contours, leafnumber,center,radius):
    circlemask = cv2.circle(np.zeros_like(necrosisprobabilityimage),center,radius,1,-1)
    necrosis_pixels = (image_with_contours==leafnumber) & (circlemask)
    return necrosis_pixels
    
def find_necrosis_radius(image_with_contours, necrosisprobabilityimage,center,leafnumber):
   minimumradius =50
   maximumradius = 1000
   interval = 30
   finalradius = 0
   oldnecrosis_pixels = np.zeros_like(necrosisprobabilityimage,dtype=bool)
   for radius in range(minimumradius, maximumradius,30):
       necrosis_pixels = find_necrosis_pixels(image_with_contours, leafnumber,center,radius)
       thistimenecrosis_pixels = (necrosis_pixels & ~ oldnecrosis_pixels)
       average_necrosis_pixels = np.average(necrosisprobabilityimage[thistimenecrosis_pixels])
       if average_necrosis_pixels>0.5:
           break
       finalradius = radius
       oldnecrosis_pixels = necrosis_pixels
   return oldnecrosis_pixels,finalradius

def find_leafcontours(image):
    # Convert the image to grayscale
    gray = np.uint8(255*image)

    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (51, 51), 0)

    # Threshold the image to create a binary image (black and white)
    _, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter_contours_by_size(contours, min_contour_area=5000)
    # Draw the contours on a copy of the original image
    image_with_contours = np.zeros_like(image)
    leafareas = list()
    for i in range(0,len(contours)):
        cv2.drawContours(image_with_contours, contours, i, i+1, thickness=cv2.FILLED)
        leafareas.append(cv2.contourArea(contours[i]))
    return contours, image_with_contours, leafareas



def filter_contours_by_size(contours, min_contour_area=100):
    filtered_contours = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area >= min_contour_area:
            filtered_contours.append(contour)
    return filtered_contours


def closestandfurthestpoints(skeleton,point):
    skeletoncoordinates = np.where(skeleton==True)
    skeletoncoordinates = np.asarray(skeletoncoordinates).T
    point = point[::-1]
    Distances = np.sum((skeletoncoordinates-point)**2,axis=1)
    closestpoint = np.argmin(Distances)
    closestpoint = skeletoncoordinates[closestpoint,:]
    closestdistance = math.sqrt(np.min(Distances))
    furthestpoint = np.argmax(Distances)
    furthestpoint = skeletoncoordinates[furthestpoint,:]
    furthestdistance = math.sqrt(np.max(Distances))
    
    return closestpoint,furthestpoint,closestdistance,furthestdistance

# Draws outlines of the plug,petridish,leaf and necrosis in the original image
def drawOutlines(impath,outpath,leafimage,necrosisimage,petriimage,plugimage):
    im = h5py.File(impath)
    im = np.squeeze(im['data'][0])
    im = im[:,:,::-1]
    im = np.ascontiguousarray(im, dtype=np.uint8)
    #im = cv2.imread(impath)
    # if len(im) == 3024:
    im = cv2.rotate(im, cv2.cv2.ROTATE_90_CLOCKWISE)
    #cv2.imshow('image',im)
    contours,_ = cv2.findContours(np.flip(np.uint8(leafimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i in contours:
        cv2.drawContours(im,i,-1,(0,255,0),thickness=5)
    contours,_ = cv2.findContours(np.flip(np.uint8(necrosisimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i in contours:
        cv2.drawContours(im,i,-1,(0,0,0),thickness=5)
    contours,_ = cv2.findContours(np.flip(np.uint8(petriimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i in contours:
        cv2.drawContours(im,i,-1,(0,0,255),thickness=5)
    contours,_ = cv2.findContours(np.flip(np.uint8(plugimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i in contours:
        cv2.drawContours(im,i,-1,(255,0,0),thickness=5)        
    cv2.imwrite(outpath,im)
    return

def plugfinder(leafimage,plugprobabilityimage):
    radius = 50
    plugcentredict = dict()
    plugleafdict = dict()
    blur = cv2.GaussianBlur(plugprobabilityimage,(radius*2+1,radius*2+1),radius)
    plugimage = np.zeros(np.shape(leafimage))
    leafindexes = np.unique(leafimage[leafimage!=0])
    for i in leafindexes:
        minvalue = np.min(blur[leafimage==i])
        centre = np.where(blur==minvalue)
        plugcentredict[int(i)]=[int(centre[1][0]),int(centre[0][0])]
        plugleafdict[int(i)]=int(i)
        plugimage = cv2.circle(plugimage,(centre[1][0],centre[0][0]),radius,int(i),-1)
    return plugimage,plugcentredict,plugleafdict

def petrifinder(petriprobabilityimage):
    petriuint8 = cv2.blur(petriprobabilityimage, (50, 50))
    petriuint8 = np.uint8(petriuint8*255)
    detected_circles = cv2.HoughCircles(petriuint8,cv2.HOUGH_GRADIENT, 1.5, 3000, param1 = 50,param2 = 30, minRadius = 1000, maxRadius = 2000)
    petriimage = np.zeros(np.shape(petriuint8))
    a = int(detected_circles[0][0][0])
    b = int(detected_circles[0][0][1])
    r = int(detected_circles[0][0][2])
    petriimage = cv2.circle(petriimage,(a,b),r,1,-1)
    petriarea = np.pi*r**2
    return petriimage,[a,b],r,petriarea
    
    
# Uses the average necrosis pixel probability in a leaf segment to monitor spread 
def necrosisfinder(leafimage,necrosisprobabilityimage,leafnumber,necrosismask=0,manual=[]):
    originalnecrosismask = necrosismask.copy()
    threshold = 0.5
    contours = cv2.findContours(np.uint8(leafimage==leafnumber),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
    ellips= cv2.fitEllipse(contours[0][0])
    image_center = ellips[0]
    angle = ellips[2]
    global rot_mat, rot_matinv
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_matinv = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
    result = cv2.warpAffine(np.uint8(leafimage==leafnumber), rot_mat, leafimage.shape[1::-1], flags=cv2.INTER_LINEAR)
    try:
        minvalue = np.min(np.where(result==1)[0])
        maxvalue = np.max(np.where(result==1)[0])
        if len(manual)>0:
            startpoint = manual[0]
            endpoint = manual[1]
            startcoords = np.dot(rot_mat[0:2,0:2],startpoint)+rot_mat[:,2]
            endcoords = np.dot(rot_mat[0:2,0:2],endpoint)+rot_mat[:,2]
            start = int(np.max([0,np.min([startcoords[1],endcoords[1]])]))
            end = int(np.max([startcoords[1],endcoords[1]]))
            result[start:end]=2
            resultrot = cv2.warpAffine(result,rot_matinv, leafimage.shape[1::-1], flags=cv2.WARP_FILL_OUTLIERS)   
            resultrot[leafimage!=leafnumber]=0
            resultrot[resultrot==1]=0
            resultrot[resultrot==2]=1
            necrosismask = resultrot
        else:
            segments = range(minvalue,maxvalue,int((maxvalue-minvalue)/50))
            segmentmask = np.zeros(np.shape(leafimage))
            coords = np.where(result==1)
            for i in range(1,len(segments)):
               segmentcoords = (coords[0][coords[0]>=segments[i-1]],coords[1][coords[0]>=segments[i-1]])
               segmentmask[segmentcoords]=i
            segmentmaskrot = cv2.warpAffine(segmentmask,rot_matinv, leafimage.shape[1::-1], flags=cv2.WARP_FILL_OUTLIERS)
            necrosislist = list()
            goodtogo = 0
            strictthreshold = 0.8
            for i in range(1,len(segments)):
                rotcoords = (segmentmaskrot==i)
                avnecrosis = np.mean(necrosisprobabilityimage[rotcoords])
                if avnecrosis > threshold:
                    necrosismask[rotcoords]=1
                    necrosislist.append(i)
                if avnecrosis> strictthreshold:
                    goodtogo = 1
            if goodtogo == 0:
                necrosislist = list()
                necrosismask = originalnecrosismask
            for i in range(1,len(segments)):
                if i not in necrosislist:
                    if (i+1 in necrosislist and i-1 in necrosislist) or (i+2 in necrosislist and i-2 in necrosislist):
                        necrosislist.append(i)
                        rotcoords = (segmentmaskrot==i)
                        necrosismask[rotcoords]=1
            necrosismask = measure.label(necrosismask)
    except ValueError as err: 
        necrosismask = np.zeros(np.shape(leafimage))
    return necrosismask
