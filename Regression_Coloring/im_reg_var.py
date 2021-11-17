
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from PIL import Image
from pandas.core.base import GroupByError
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from numpy import asarray
from MatrixOpLibrary import scaled_mat
from Image_Applications import colorTogrey


# use data set of 101 images, select 1 at random to color grey, then use remaining 100 to attempt to color it 
# split into training and testing sets 80/20
# convert to png if needed, this is for compatibility with image applications/matrix libraries
# https://stackoverflow.com/questions/10759117/converting-jpg-images-to-png
# https://www.diva-portal.org/smash/get/diva2:519159/fulltext01.pdf

# -------------------------Coloring Attempt with multiple images--------------------------------#
#randomly select a random image to color grey out of the file
#loop through remaining 101 comic faces 
#if .jpg then convert to png 
#get reds, blues, and greens and append to reds, blues, and greens arrays to store in df
jpglist = [f'{i}.jpg' for i in range(101)]
i = 1
for face in jpglist:
    jpg = Image.open('ComicFaces/' + face)
    jpg.save(f'PNGS/face{i}.png')
    i = i + 1

# check if all images are the same size and scale them down by a factor of 3 
pnglist = [f'face{i}.png' for i in range(1,102)]
j=1
for face in pnglist:
    png = asarray(plt.imread('PNGS/' + face))
    scaled_png = scaled_mat(png,3)
    plt.imsave(f'Scaled_PNGS/{j}.png',scaled_png)
    j = j + 1

# pick a random scaled image and color it grey, then use all other pictures to collect RGB data
scaled_pnglist = [f'{i}.png' for i in range(1,102)]
random_num = random.randint(1,102)
random_grey_image = asarray(plt.imread('Scaled_PNGS/'+ scaled_pnglist[random_num]))
grey_face = colorTogrey(255*random_grey_image)
plt.imsave('Scaled_PNGS/face_to_color.png',grey_face)

all_reds = []
all_greens = []
all_blues = []
without_grey = [f'{i}.png' for i in range(1,102) if i != random_num]
m_,n_ = len(grey_face),len(grey_face[0])

for image in without_grey:
    imageMat = asarray(plt.imread('Scaled_PNGS/' + image))
    rs = [imageMat[i][j][0] for i in range(m_) for j in range(n_)]
    gs = [imageMat[i][j][1] for i in range(m_) for j in range(n_)]
    bs = [imageMat[i][j][2] for i in range(m_) for j in range(n_)]
    all_reds = all_reds + rs
    all_greens = all_greens + gs
    all_blues = all_blues + bs

coords = [(x,y) for x in range(m_) for y in range(n_)]
all_pics_rgb = pd.DataFrame({'Xs':len(without_grey)*[c[0] for c in coords],'Ys':len(without_grey)*[c[1] for c in coords],'Reds':all_reds,'Greens':all_greens,'Blues':all_blues})

# in order to convert this data frame into the same size of the image we are trying to color, 
# need to run a regression per picture, then average out by (x,y) location, then apply it to the 
# grey image 
poly = PolynomialFeatures(degree = 2)
clf = LinearRegression()
step = 0
prediction_images = []
for picture in range(len(without_grey)):
    per_pic_df = all_pics_rgb[step*len(coords):(step+1)*len(coords)]
    allX_= poly.fit_transform(per_pic_df[['Xs','Ys']])
    clf.fit(allX_, per_pic_df['Reds'])
    per_pic_predicted_reds = clf.predict(poly.fit_transform(per_pic_df[['Xs','Ys']]))
    clf.fit(allX_, per_pic_df['Greens'])
    per_pic_predicted_greens = clf.predict(poly.fit_transform(per_pic_df[['Xs','Ys']]))
    clf.fit(allX_, per_pic_df['Blues'])
    per_pic_predicted_blues = clf.predict(poly.fit_transform(per_pic_df[['Xs','Ys']]))
    new_polynomial_rgbs = list(zip(per_pic_predicted_reds,per_pic_predicted_greens,per_pic_predicted_blues))
    mat = asarray([new_polynomial_rgbs[n_*i:n_*(i+1)] for i in range(m_)])

    per_pic_rgbs_poly = [[0 for x in range(n_)] for x in range(m_)]
    for i in range(m_):
        for j in range(n_):
            per_pic_rgbs_poly[i][j] = np.append(mat[i][j],1)
    final_mat = asarray(per_pic_rgbs_poly)
    prediction_images.append(final_mat)
    step = step + 1 

for image in prediction_images:
    c = [image[i][j] for i in range (len(image)) for j in range(len(image[0]))]

# -------------------------Coloring Attempt with One Image--------------------------------#
im = plt.imread('Output_Images/girl.png')
im_mat = asarray(im)
A = scaled_mat(im_mat,4)

im_to_color = plt.imread('Output_Images/test.png')
C = asarray(im_to_color)

# make sure colored images and grey image that will be colored are the same size

# https://stats.stackexchange.com/questions/58739/polynomial-regression-using-scikit-learn
m = len(A)
n = len(A[0])

# create a dataframe that stores each locations rgb values ( this will be used for all parts )
# will then implement this to store many more images for much larger df
reds = [A[i][j][0] for i in range(m) for j in range(n)]
greens = [A[i][j][1] for i in range(m) for j in range(n)]
blues = [A[i][j][2] for i in range(m) for j in range(n)]
coords = [(x,y) for x in range(m) for y in range(n)]
coords_df = pd.DataFrame({'Xs':[c[0] for c in coords],'Ys':[c[1] for c in coords],'Reds':reds,'Greens':greens,'Blues':blues})

# -------------------------Coloring Attempt with Linear Regression--------------------------------#

# define model that will be used and perform regression on each color channel

ols = LinearRegression()
num_observations = len(coords)
# regression on reds
ols.fit(coords_df[['Xs','Ys']], coords_df['Reds'])
coefs_reds = ols.coef_
ints_reds = ols.intercept_

# regression on greens
ols.fit(coords_df[['Xs','Ys']], coords_df['Greens'])
coefs_greens = ols.coef_
ints_greens = ols.intercept_

# regression on blues
ols.fit(coords_df[['Xs','Ys']], coords_df['Blues'])
coefs_blues = ols.coef_
ints_blues = ols.intercept_

# print(coefs_reds,coefs_greens,coefs_blues)
# print(ints_reds,ints_greens,ints_blues)

# define trends line for colors as dependent on coordinates
def red_line(x,y): return ints_reds + coefs_reds[0]*x + coefs_reds[1]*y 
def green_line(x,y): return ints_greens + coefs_greens[0]*x + coefs_greens[1]*y 
def blue_line(x,y): return ints_blues + coefs_blues[0]*x + coefs_blues[1]*y 

# based on modeled values, input coordinates of picuture that will be colored to compute 
# supposed color at each location based on the training set (soon will be n pictures not just one)
xcords = [x for x in range(n)]
ycords = [y for y in range(m)]
new_reds = [red_line(x,y) for x in xcords for y in ycords]
new_greens = [green_line(x,y) for x in xcords for y in ycords]
new_blues = [blue_line(x,y) for x in xcords for y in ycords]
new_rgbs = list(zip(new_reds,new_greens,new_blues))
rgbs_as_mat = asarray([new_rgbs[n*i:n*(i+1)] for i in range(n)])

rgbs_plus_channel = [[0 for x in range(n)] for x in range(m)]

for i in range(n):
    for j in range(n):
        rgbs_plus_channel[i][j] = np.append(rgbs_as_mat[i][j],1)
final_rgbs_mat = asarray(rgbs_plus_channel)

# color the new picture based on newly found colors
# picture to be color will be grey, so need to keep grey color as part of implentation as that will 
# determine the intensity of the pixel based on given RGB values from model
# color the image 

for row in range(m):
    for col in range(n):
        C[row][col] = asarray(C[row][col])*asarray(final_rgbs_mat[row][col])

# a linear regression basically gave a line with little slope, meaning the colors were simply constant
# the fact that the y-intercept was in the range [.4,.6] means its around the average of the rgb values, or brown/tan
# the result was a tan coloring of the original image
# plt.imsave('Output_Images/colored_image_linear_regression.png',C)

# -------------------------Coloring Attempt with Polynomial Regression--------------------------------#
B = C
poly = PolynomialFeatures(degree = 4)
X_ = poly.fit_transform(coords_df[['Xs','Ys']])
clf = LinearRegression()

clf.fit(X_, coords_df['Reds'])
predicted_reds = clf.predict(poly.fit_transform(coords_df[['Xs','Ys']]))

clf.fit(X_, coords_df['Greens'])
predicted_greens = clf.predict(poly.fit_transform(coords_df[['Xs','Ys']]))

clf.fit(X_, coords_df['Blues'])
predicted_blues = clf.predict(poly.fit_transform(coords_df[['Xs','Ys']]))

new_polynomial_rgbs = list(zip(predicted_reds,predicted_greens,predicted_blues))
polynomial_rgbs_as_mat = asarray([new_polynomial_rgbs[n*i:n*(i+1)] for i in range(n)])

# **************original incorrect attempt but correct plot******************
# clf.fit(X_, coords_df['Blues'])
# plt.plot(coords_df[['Xs','Ys']],clf.predict(poly.fit_transform(coords_df[['Xs','Ys']])), color  = 'blue')
# def b_line(x,y): return clf.coef_[0]+clf.coef_[1]*x+clf.coef_[2]*y+clf.coef_[3]*x*y+clf.coef_[4]*(x**2)+clf.coef_[5]*(y**2) 
# new_blues_poly = [b_line(x,y) for x in xcords for y in ycords]
# ***************************************************************************

# add color channel to multiply vectors with original image vectors
rgbs_poly = [[0 for x in range(n)] for x in range(m)]

for i in range(m):
    for j in range(n):
        rgbs_poly[i][j] = np.append(polynomial_rgbs_as_mat[i][j],1)
final_rgbs_mat_poly = asarray(rgbs_poly)

# color the new picture based on newly found colors
# picture to be color will be grey, so need to keep grey color as part of implentation as that will 
# determine the intensity of the pixel based on given RGB values from model
# color the image 
for row in range(m):
    for col in range(n):
        B[row][col] = asarray(B[row][col])*asarray(final_rgbs_mat_poly[row][col])
        #print(B[row][col])

# result was that the colors in the colored image were in fact modeled better than linear regression. The face was lighter and 
# the background was darker. However the colors being applied are still fairly similair with a brown shade. Certain degrees when 
# choosing an n degree polynomial result in negative rgb values.
# plt.imsave('Output_Images/colored_image_polynomial_regression.png',B)
