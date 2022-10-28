import numpy as np
import cv2 as cv
import math

def binary_thresh(img, color):
    """
    img : The input image matrix
    color : A string for the type of color we want (hardcoded). Options include : "yellow" , "white", "red"
    ---
    Returns thresh : Binary thresholded image according to the color value
    """

    # The matrix we will return
    thresh = np.zeros(img.shape)
        
    # all bgr values outside color values are turned to black, otherwise they become white
    if color == "yellow":
        thresh = cv.inRange(img, (0, 200, 200), (50, 255, 255))

    if color == "white":
        thresh = cv.inRange(img, (180, 180, 180), (255, 255, 255))

    if color == "red":

        thresh = cv.inRange(img, (0, 0, 150), (120, 120, 255))

    if color == "green":
        thresh = cv.inRange(img, (0,140,0),(200,255,100))

    if color == "blue":
        thresh = cv.inRange(img, (180,0,0), (255,50,50))


    return thresh

def top_mask(img, mask_h, mask_w, Trapezoid = True):
    """
    Creates a mask that covers the topside of the image. The masks shape is trapezoidal if there is an angle argument
    ---
    img : ndarray, input image
    mask_h : int, the height of the mask (higher the value, bigger the mask).
             The mask will cover from top of image until mask_h point
    mask_w : int, the width of the mask (higher the value, bigger the mask).
             The mask will cover from beginning (left) of image until mask_w point.
    Trapezoid : bool, whether or not mask is trapezoidal shaped or square shaped
    ---
    Returns : ndarray, masked image
    """

    # color to grayscale
    # gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    x = img.shape[1]
    y = img.shape[0]

    # normal mask if no mask_w argument
    if mask_w == 0:

        # if trapezoid argument is true, we change mask accordingly
        if Trapezoid:
            topl = [int(4*x/10), int(mask_h)]
            topr = [int(6*x/10),int(mask_h)]
            midl = [0,y-int(mask_h/8)]
            midr = [x,y-int(mask_h/8)]
        else :
            topl = [0, int(mask_h)]
            topr = [x, int(mask_h)]
            midl = [0,y-int(mask_h/2)]
            midr = [x,y-int(mask_h/2)]


        botl = [0,y]
        botr = [x,y]


    # left starting points = mask_w if there is a mask_w argument
    else :
        
        # if trapezoid argument is true, we change mask accordingly
        if Trapezoid:
            topl = [mask_w, int(mask_h)]
            topr = [int(6*x/10),int(mask_h)]
            midl = [mask_w,y-int(mask_h/8)]
            midr = [x,y-int(mask_h/8)]
        else :
            topl = [mask_w, int(mask_h)]
            topr = [x, int(mask_h)]
            midl = [mask_w,y-int(mask_h/2)]
            midr = [x,y-int(mask_h/2)]

        botl = [mask_w,y]
        botr = [x,y]

    # create the polygon and the mask
    polygon = np.array([topl,topr, midr, botr, botl, midl], dtype = 'int32')
    mask = np.zeros(img.shape, dtype = 'int32')
    cv.fillPoly(mask, [polygon], color = 255)


    # change all values outside of mask shape to black
    mask = cv.bitwise_and(mask.astype('uint8'), img)

    return mask

def find_white_ind(img, row, sensitivity, order):
    """
    Takes a row and finds first or last white pixel depending on the order argument.
    Does this for all rows above 'row' ([row - sensitivity, row]) and returns the average point
    ---
    img : ndarray, input image
    row : int, row where we will try to find the last pixel.
    sensitivity : int, number of rows above 'row'. Will take their average value.
    order : string, last or first.
    ---
    Returns : int, y coordinate of average point if it is found
            : -1, if it is not found
    """

    # the starting index of the range where we will check for white pixels
    y_start = row - sensitivity
    y_end = row
    
    # Array to store the last/first white pixel index value for each row
    ind = []

    for i in range(y_start,y_end):

        # horizontal line (row) of an image
        line = img[i,:]

        # find the indices of white pixels
        found_indices = np.argwhere(line == 255)

        # if we find a pixel in the line (it is not empty)
        if (found_indices.size!=0):
            
            if order == "last":

                # find the last occurence of 255 (white)
                max = np.max(found_indices)
                ind.append(max)

            if order ==  "first":
                
                # find the first occurence of 255 (white)
                min = np.min(found_indices)
                ind.append(min)

    # if the array of last/first indices is not empty we return the mean
    if (len(ind)>0):

        # find the mean value
        # mean = np.mean(ind)
        mean = np.median(ind)
        return mean

    # else we return a -1 to signify an error value
    else:
        return -1

def moments(img):
    """
    Calculate moments (centroid) of a shape. Inspired by : https://learnopencv.com/tag/cv2-moments/
    ---
    img : ndarray, input image
    ---
    Returns
    cX, cY : x and y coordinates of moment
    """

    
    # calculate moments
    M = cv.moments(img)

    if (M["m00"])!= 0:
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    else: 
        # calculate x,y coordinate of center
        cX = int(M["m10"] / (M["m00"] + 0.0001))
        cY = int(M["m01"] / (M["m00"] + 0.0001))


    return cX, cY

def create_lane_polygon(img, left_color, right_color):
    """
    Takes an image with of lane,
    returns an array of polygon points that frame the lane
    ---
    img : ndarray, input image
    left_color : string, color of left line of lane
    right_color : string, color of right line of lane
    ---
    Returns
    polygon : ndarray, polygon points
    """

    # image variables
    height = img.shape[0]
    width = img.shape[1]

    # mask variables    
    mask_h = int(3*height/5) # heigth
    mask_w = int(3*width/5) # width

    # ==================== YELLOW / WHITE MASK ======================= #

    # binary threshold where left lines will be the white component
    left_thresh = binary_thresh(img,left_color)
    left_mask = top_mask(left_thresh, mask_h, 0, Trapezoid = True)

    # binary threshold where white lines will be the white component
    right_thresh = binary_thresh(img,right_color)
    right_mask = top_mask(right_thresh , mask_h, mask_w, Trapezoid = True)

    # ================= POLYGON PARAMETERS =================== #

    
    jump = 4 # number of pixels we skip between each polygon point
    color = (0,255,0) # polygon color
    thickness = 3 # polygon thickneses


    # ================= LEFT POINTS OF POLYGON =============== #
    left = []

    # Find the last white pixel in image for each row starting from 'mask_h' row
    for i in range(mask_h, height, jump):

        l = find_white_ind(left_mask, i, jump, "last")
        if l != -1:
            left.append([l,i])

    # ================= RIGHT POINTS OF POLYGON =============== #

    right = []
    
    # Find the first white pixel in image for each row starting from 'mask_h' row
    for i in range(height, mask_h, -jump):

        r = find_white_ind(right_mask, i, jump, "first")
        if r != -1:
            right.append([r,i])

    # =================== POLYGON CREATION ===================== #
    # left = left + [[0,height-1]]
    # Polygon points concatenated
    polygon = left + right
    polygon = np.array(polygon, dtype = 'int32')

    return polygon

def create_lane_polygon_right(img, right_color):
    """
    Takes an image of lane,
    returns an array of polygon points that frame the lane
    ---
    img : ndarray, input image
    right_color : string, color of right line of lane
    ---
    Returns
    polygon
    """

    # image variables
    height = img.shape[0]
    width = img.shape[1]

    # mask variables    
    mask_h = int(3*height/5) # heigth
    mask_w = int(3*width/5) # width

    # ==================== RIGHT LINE MASK ======================= #

    # binary threshold where white lines will be the white component
    right_thresh = binary_thresh(img,right_color)
    right_mask = top_mask(right_thresh , mask_h, mask_w, Trapezoid = True)

    # ================= POLYGON PARAMETERS =================== #

    
    jump = 4 # number of pixels we skip between each polygon point
    color = (0,255,0) # polygon color
    thickness = 3 # polygon thickneses

    # ================= RIGHT POINTS OF POLYGON =============== #

    right = []

    # Find the first white pixel in image for each row starting from 'mask_h' row
    for i in range(height, mask_h, -jump):

       
        r = find_white_ind(right_mask, i, jump, "first")
        if r != -1:
            right.append([r,i])

    right = np.array(right)

    # ================= LEFT POINTS OF POLYGON =============== #

    left = np.array(right)

    # fill array of left side of polygon, it will mirror the right line
    # but shifted to the left
    left[:,0] = right[:, 0] - int(width/2)

    left = np.flip(left, axis = 0 )

    # =================== POLYGON CREATION ===================== #

    # Polygon points concatenated
    polygon = np.concatenate((left, right))
    polygon = polygon.astype('int32')

    return polygon

def create_lane_polygon_real(img, right_color):

    # image variables
    height = img.shape[0]
    width = img.shape[1]

    # mask variables    
    mask_h = int(2*height/5) # heigth
    mask_w = int(3*width/5) # width

    # ==================== IMAGE PROCESSING ====================== #

    # red channel of image
    #red_img = img[:,:,2]
    # cv.imshow("RED CHANNEL", red_img)

    # horizontal gradient
    #red_sobel = sobel_mod(red_img, 'x', 5)
    # cv.imshow("SOBEL", red_sobel)

    # thresholding
    #ret, thresh = cv.threshold(red_sobel,20,255,cv.THRESH_BINARY)
    # cv.imshow("THRESH", thresh)

    # remove blur by using morph opening
    #kernel = np.ones((3,3), np.uint8)
    # opening = thresh
    #opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    # opening = cv.erode(thresh, kernel, iterations = 1)
    # cv.imshow("OPEN", opening)

    

    # ==================== RIGHT LINE MASK ======================= #

    # binary threshold where white lines will be the white component
    right_thresh = binary_thresh(img,right_color)
    # right_mask = top_mask(opening , mask_h, mask_w, Trapezoid = False)
    right_mask = top_mask(right_thresh , mask_h, mask_w, Trapezoid = False)

    cv.imshow("RIGHT", right_mask)

    # ================= POLYGON PARAMETERS =================== #

    
    jump = 10 # number of pixels we skip between each polygon point
    color = (0,255,0) # polygon color
    thickness = 3 # polygon thickneses

    # ================= RIGHT POINTS OF POLYGON =============== #

    right = []

    for i in range(height, mask_h, -jump):

        r = find_white_ind(right_mask, i, jump, "first")
        if r != -1:
            right.append([r,i])

    right = np.array(right)

    # ================= LEFT POINTS OF POLYGON =============== #

    left = np.array(right)

    # fill array of left side of polygon, it will mirror the right line
    # but shifted to the left
    if (right.size!=0):

        left[:,0] = right[:, 0] - int(width/2)

    left = np.flip(left, axis = 0 )

    # =================== POLYGON CREATION ===================== #

    # left = left + [[0,height-1]]
    # Polygon points concatenated
    polygon = np.concatenate((left, right))
    polygon = polygon.astype('int32')

    return polygon

def create_lane_polygon_real2(img, left_color, right_color):
    """
    Takes an image with of lane,
    returns an array of polygon points that frame the lane
    ---
    img : ndarray, input image
    left_color : string, color of left line of lane
    right_color : string, color of right line of lane
    """

    # image variables
    height = img.shape[0]
    width = img.shape[1]

    # mask variables    
    mask_h = int(3*height/5) # heigth
    mask_w = int(3*width/5) # width

    # ==================== YELLOW / WHITE MASK ======================= #

    # binary threshold where left lines will be the white component
    left_thresh = binary_thresh(img,left_color)
    left_mask = top_mask(left_thresh, mask_h, 0, Trapezoid = True)

    # binary threshold where white lines will be the white component
    right_thresh = binary_thresh(img,right_color)
    right_mask = top_mask(right_thresh , mask_h, mask_w, Trapezoid = True)

    # ================= POLYGON PARAMETERS =================== #

    
    jump = 4 # number of pixels we skip between each polygon point
    color = (0,255,0) # polygon color
    thickness = 3 # polygon thickneses


    # ================= LEFT POINTS OF POLYGON =============== #
    left = []

    # fill array of left side of polygon
    for i in range(mask_h, height, jump):

        l = find_white_ind(left_mask, i, jump, "last")
        if l != -1:
            left.append([l,i])

    # ================= RIGHT POINTS OF POLYGON =============== #

    right = []

    for i in range(height, mask_h, -jump):

        r = find_white_ind(right_mask, i, jump, "first")
        if r != -1:
            right.append([r,i])

    # =================== POLYGON CREATION ===================== #
    # left = left + [[0,height-1]]
    # Polygon points concatenated
    polygon = left + right
    polygon = np.array(polygon, dtype = 'int32')

    return polygon

def draw_lane_polygon(img, polygon):

    """
    Takes polygon points and draws the lane mask
    ---
    img : ndarray, input image
    polygon : ndarray, polygon points
    ---
    Returns
    lane : ndarray, image with lane mask
    """
    # image variables
    height = img.shape[0]
    width = img.shape[1]

    # draw polygon
    lane = np.array(img)
    lane = cv.polylines(lane, [polygon], isClosed = True, color = (0,255,0), thickness = 3)
    
    # find its center
    center = moments(polygon) # Center of polygon 

    # draw a circle at the center
    radius = 5 # Radius of circle
    color = (255, 0, 0) # Color in BGR
    thickness = 2 # Line thicknes
    lane = cv.circle(lane, center, radius, color, thickness)

    # center line points
    pt1 = (int(width/2), int(5*height/6))
    pt2 = (int(width/2), height)

    # draw a vertical red line at the bottom of the image around the center
    radius = 5 # Radius of circle
    color = (255, 0, 0) # Color in BGR
    thickness = 2 # Line thicknes    # draw center line
    cv.line(lane, pt1,pt2, color = (0,0,255), thickness = thickness)

    return lane

def check_for_red(img):
    """
    Checks if there are red pixels around the bottom of the image
    """
    height = img.shape[0]
    width = img.shape[1]

    red_mask = binary_thresh(img, "red")

    # if a white pixel exists in bottom center of the image return True
    if 255 in red_mask[int(9*height/10): height, int(width/2)]:
        
        return True
    
    # if it doesn't return false
    else:
        return False

def sobel_mod(img, grad, ksize):
    """
    Modified version of sobel filter
    ---
    img : ndarray, the input image matrix
    grad : char, direction of the sobel gradient (either x or y)
    ksize : int, size of the filter kernel
    ---
    Returns sobel
    """

    # color to grayscale
    # gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

  
    ddepth = cv.CV_16S # image output depth

    if grad == 'x':

        # sobel with sideways gradien
        sobel = cv.Sobel(img, ddepth, 1, 0, ksize=3, borderType=cv.BORDER_DEFAULT)

    elif grad == 'y':
        # sobel with sideways gradien
        sobel = cv.Sobel(img, ddepth, 0, 1, ksize=3, borderType=cv.BORDER_DEFAULT)

    # # converting back to uint8
    sobel = cv.convertScaleAbs(sobel)

    return sobel

def hough_mod(img):

    # cloned img but where we will draw hough lines
    h_image = np.array(img)

    rows = h_image.shape[0]
    cols = h_image.shape[1]

    # # color to grayscale
    # gray = cv.cvtColor(h_image,cv.COLOR_BGR2GRAY)

    # edge detection
    edges = cv.Canny(img,50,150,apertureSize = 3)

    # hough parameters
    minLineLength = 30
    maxLineGap = 10

    lines = cv.HoughLines(edges,1,np.pi/180,100)

    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):

            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(h_image, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

    return h_image


