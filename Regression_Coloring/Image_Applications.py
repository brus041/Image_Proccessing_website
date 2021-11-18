import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
from MatrixOpLibrary import matmat,matvec,scaled_mat,id_mat,rotation_mat_2d
from math import ceil

image =Image.open("Output_Images/girl.png")
#image = (plt.imread("Output_Images/girl.png"))
#scaled_image = scaled_mat(image,4)
#-----------Set each pixel color equal to the average color values of its neighbors------------------

def average_coloring(im,scale_fac,nghbrs):
    img = im
    #img = asarray(im)
    (mG, nG, dimG) = img.shape

    # compress_image
    # Shrink dimension of image (original to half or chosen n)

    compress_factor = scale_fac
    scaling_factor = 1/compress_factor
    new_img_cols = [row[::compress_factor] for row in img]
    new_img_rows = asarray([new_img_cols[i] for i in range(0, len(img), compress_factor)])

    # blur image by setting each pixel to be the average of its neighboring pixels
    # (this created a coloring affect rather than a blur)
    # dimensions
    
    m = len(img)-1
    n = len(img[0])-1
    if nghbrs == 'adjacent':
        # corner cases (divide by 2 neighbors)
        img[0][0] = (img[1][0] + img[0][1]) / 2
        img[m][n] = (img[m - 1][n] + img[m][n - 1]) / 2
        img[m][0] = (img[m - 1][0] + img[m][1]) / 2
        img[0][m] = (img[0][m - 1] + img[1][m]) / 2

        # Edge column and row cases without corners (divide by 3 neighbors)
        for i in range(m):
            for j in range(n):
                img[0][j] = (img[0][j-1]+img[0][j+1]+img[i+1][j])/3
                img[0][j][3] = 255

                img[m][j] = (img[m][j-1]+img[m][j+1]+img[m-1][j])/3
                img[m][j][3] = 255

                img[i][0] = (img[i-1][0]+img[i+1][0]+img[i][j+1])/3
                img[i][0][3] = 255

                img[i][n] = (img[i-1][n]+img[i+1][n]+img[i][n-1])/3
                img[i][n][3] = 255

        # Pixels inside of bordering rows and columns (divide by 4 neighbors)
        for i in range(1, n-1):
            for j in range(1, m-1):
                img[i][j] = (img[i][j-1]+img[i][j+1]+img[i-1][j]+img[i+1][j])/4
                img[i][j][3] = 255

        return img

    elif nghbrs == "all":
        # corner cases (divide by 3 neighbors)
        img[0][0] = (img[1][0] + img[0][1]+ img[1][1]) / 3
        img[m][n] = (img[m - 1][n] + img[m][n - 1]+img[m-1][n-1]) / 3
        img[m][0] = (img[m - 1][0] + img[m][1]+img[m-1][1]) / 3
        img[0][m] = (img[0][m - 1] + img[1][m]+img[1][m-1]) / 3

        # Edge column and row cases without corners (divide by 5 neighbors)
        for i in range(m):
            for j in range(n):
                img[0][j] = (img[0][j-1]+img[0][j+1]+img[1][j]+img[1][j+1]+img[1][j-1])/5
                img[0][j][3] = 255

                img[m][j] = (img[m][j-1]+img[m][j+1]+img[m-1][j]+img[m-1][j-1]+img[m-1][j+1])/5
                img[m][j][3] = 255

                img[i][0] = (img[i-1][0]+img[i+1][0]+img[i][1]+img[i-1][1]+img[i+1][1])/5
                img[i][0][3] = 255

                img[i][n] = (img[i-1][n]+img[i+1][n]+img[i][n-1]+img[i+1][n-1]+img[i-1][n-1])/5
                img[i][n][3] = 255

        # Pixels inside of bordering rows and columns (divide by 8 neighbors)
        for i in range(1, n-1):
            for j in range(1, m-1):
                img[i][j] = (img[i][j-1]+img[i][j+1]+img[i-1][j]+img[i+1][j]
                + img[i-1][j-1]+img[i+1][j+1]+img[i-1][j+1]+img[i+1][j-1])/8
                img[i][j][3] = 255
        return img
    
    else:
         return img
# op = average_coloring(image,2,'all')
# print(op[0])
img1 = np.array(Image.open("Output_Images/girl.png"))
plt.imsave("av_image.png", average_coloring(img1,2,'adjacent'))

#--------------Multiply an image by Identity Matrix to get unchanged image-----------------------
#--------------Multiply an image by Identity Matrix with alpha !=1 to get brighter Image---------

#Compressed Image x Identity Matrix
# image_mat = asarray(image)
# scaled_image = scaled_mat(image_mat,4)

# cN, cM = len(scaled_image),len(scaled_image[0])
# n0 = min([cN,cM])
# id_n = id_mat(n0,1)
# matprod_ID_Image = matmat(scaled_image, id_n)
# plt.imsave("IDtransform.png", matprod_ID_Image)

#Compressed Image x Matrix with diagonal entries = 2
#all colors now outside interval [0,255] so image will be much brighter in appearance
# cNa, cMa = len(scaled_image),len(scaled_image[0])
# nA = min([cNa,cMa])
# id_nA = id_mat(nA,2)
# matprod_alpha_Image = matmat(scaled_image, id_nA)

# plt.imsave("ALPHAtransform.png", matprod_alpha_Image)

#--------------Multiply each image pixel by greyscale matrix to convert to greyscale------------------

def colorTogrey(im):
    
    #compress image to speed up process then
    #find dimensions of compressed image 
    img = asarray(im)
    cN, cM = len(img),len(img[0])
    n = min([cN,cM])
    
    # initialize grey matrix and use matrix-vector multiplication to alter each coordinate in the image
    grey_Mat = [[.2989, 0, 0, 0],
                [0, .5870, 0, 0],
                [0, 0, .1140, 0],
                [0, 0, 0, 1]]

    grey_IM = [[0 for x in range(n)] for x in range(n)]
    for i in range(n):
        for j in range(n):
            grey_val = sum(matvec(asarray(grey_Mat), img[i][j]))-255
            grey_IM[i][j] = [grey_val, grey_val, grey_val,255]

    return (1/255)*asarray(grey_IM)

#save greyscale result to new png file
#plt.imsave("grey_im.png", colorTogrey(image))

# convert image to black and white 
def black_and_white(image):
    img = asarray(colorTogrey(255*image))
    cM, cN = len(img),len(img[0])
    intensities = [img[i][j][0] for i in range(cM) for j in range(cN)]
    average_intensity = sum(intensities)/len(intensities)

    for i in range(cM):
        for j in range(cN):
            if img[i][j][0] <= average_intensity:
                img[i][j] = [0,0,0,1]
            else:
                img[i][j] = [1,1,1,1]

    return img
#plt.imsave("bw_im.png", black_and_white(image))
#--------------Rotate image using rotation matrix and show result with matplotlib.pyplot--------------

def rotateImage(im,theta):
    img = asarray(im)
    scaled_image = scaled_mat(img,4)

    #use rotation matrix to transform coordinate in 2D
    xvals = [x for x in range(len(scaled_image))]
    yvals = [y for y in range(len(scaled_image[0]))]

    transformed_coordinates = [matvec(rotation_mat_2d(theta),[x,y]) for x in xvals for y in yvals]

    # Get set rotated_X for all rotated values of X
    # Get set rotated_Y for all rotated values of Y
    # Get corresponding color from scaled image and match with plotted coordinate

    new_xvals = [x[0] for x in transformed_coordinates]
    new_yvals = [y[1] for y in transformed_coordinates]
    colors = [scaled_image[x][y]/255 for x in range(len(scaled_image)) for y in range(len(scaled_image[0]))]

    # plot transformed image
    plt.scatter(new_xvals, new_yvals, c = colors)
    plt.show()


 #------------------Affine Gradients----------------------------
 # Create a gradual conversion of one image into another with chosen number of transitions/incriments

def affine_blend(im1, im2, incriments):
    A = asarray(im1)
    B = asarray(im2)

    # Each image is a LIST of ROWS(mO,mG) where each row contains (nO,nG) points
    # Get Dimensions of Both Images
    (mA, nA, dimA) = A.shape
    (mB, nB, dimB) = B.shape

    # check if each pixel has 4 color channels (R,G,B,Intensity)
    # if not append 255 as default intensity to each pixel in the image
    if dimA == 3:
        An = [[0 for x in range(nA)] for x in range(mA)]
        for i in range(mA):
            for j in range(nA):
                An[i][j] = np.append(A[i][j],255)
        npA = asarray(An)
    else:
        npA = A
        
    if dimB == 3:
        Bn = [[0 for x in range(nB)] for x in range(mB)]
        for i in range(mB):
            for j in range(nB):
                An[i][j] = np.append(B[i][j],255)
        npB = asarray(Bn)
    else:
        npB = B

    # Need to scale down to smallest value of the four, as cannot add blank rows to scale to largest dim
    n = min([mA, nA, nB, mB])

    scaledA = np.asarray([row[:n] for row in npA[:n]])
    scaledB = np.asarray([row[:n] for row in npB[:n]])

    # create gradient effect to show gradual change in image from one to the other
    # add number of gradient pictures as an arguement when making this into a function
    # inc is the number of increments taken to get from one picture to the next
    inc = incriments
    gradient_image_rows = []

    affine_vals = [(1 - u / inc, u / inc) for u in range(0, inc + 1)]
    cross_fade = [(u / inc * scaledA + v / inc * scaledB) for (u, v) in affine_vals]
    gradient_image_list = [list(cross_fade[i]) for i in range(0, inc + 1)]

    # Convert to list in order to concatenate , then back to numpy array to plot result
    # this creates the column effect in resulting image
    for i in range(len(gradient_image_list)):
        gradient_image_rows = gradient_image_rows + gradient_image_list[i]

    npGradient = np.asarray(gradient_image_rows)

    # Make an (inc+1)*(inc+1) square matrix out of the two pictures
    N = npGradient
    for i in range(inc):
        npGradient = np.append(npGradient, N, 1)

    # First arguement is the name of the new image file being created, second is a !numpy! array
    min_scaling_value_denominator = ceil((1 / inc) * 255) + 1
    min_scaling_value = 1 / min_scaling_value_denominator


    #save result into png file and return the created image
    plt.imsave("Affine_Faces.png", min_scaling_value*npGradient)

    return min_scaling_value * npGradient





















