import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import statistics as stat
from sklearn import datasets, linear_model

class Global():
    def __init__(this):
        this.rpre = None
        this.lpre = None

_glb = Global()


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=6):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    def remove_outliers(l):
        if len(l) < 3:
          return l == l

        med = stat.median(l)
        dev = stat.stdev(l)

        # remove outliers
        return (l > med - dev*3) & (l < med + dev*3)

    def average_line(l):
        x = [[[x1, x2] for x1,y1,x2,y2 in line] for line in l]
        y = [[[y1, y2] for x1,y1,x2,y2 in line] for line in l]
        x = np.array(x).flatten().reshape((-1,1))
        y = np.array(y).flatten()

        # Create linear regression object
        regr = linear_model.LinearRegression()

        # Train the model using the training sets
        regr.fit(x, y)

        slope = regr.coef_
        inter = regr.intercept_

        # get previous slope and intercept
        pre = _glb.rpre if slope > 0 else _glb.lpre

        if pre:
            # update with momentum
            slope = pre['slope'] * 0.8 + slope * 0.2
            inter = pre['inter'] * 0.8 + inter * 0.2

        # updated values
        updated = { 'slope': slope, 'inter': inter }

        # save updated values
        if slope > 0:
            _glb.rpre = updated
        else:
            _glb.lpre = updated

        y1 = y.min()
        x1 = (y1 - inter) / slope
        y2 = img.shape[1]
        x2 = (y2 - inter) / slope

        result = np.array([[x1,y1,x2,y2]], dtype=int)

        return result

    slopes = np.array([[(y2-y1) / (x2-x1) if x2-x1 != 0 else 1e+100 for x1,y1,x2,y2 in line] for line in lines])
    slopes = slopes.flatten()

    notflat = np.abs(slopes) > 0.3
    lines = lines[notflat]
    slopes = slopes[notflat]

    # closepre = slopes == slopes

    # if _glb.rpre and _glb.lpre:
    #     rs = _glb.rpre['slope']
    #     ls = _glb.lpre['slope']
    #     rclose = (slopes < rs * 1.25) & (slopes > rs * 0.8)
    #     lclose = (slopes > ls * 1.25) & (slopes < ls * 0.8)
    #     closepre = closepre & (rclose | lclose)
    #     # stochastic adoption
    #     closepre = closepre | (np.random.rand(len(closepre)) < 0.1)

    # lines = lines[closepre]
    # slopes = slopes[closepre]

    positive = slopes > 0
    rlines = lines[positive]
    rslopes = slopes[positive]
    llines = lines[~positive]
    lslopes = slopes[~positive]

    lines = []

    if len(rslopes):
        rlines = rlines[remove_outliers(rslopes)]
        rline = average_line(rlines)
        lines.append(rline)

    if len(lslopes):
        llines = llines[remove_outliers(lslopes)]
        lline = average_line(llines)
        lines.append(lline)

    lines = np.array(lines)

    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


fnames = [
   'solidWhiteCurve.jpg',
   'solidWhiteRight.jpg',
   'solidYellowCurve.jpg',
   'solidYellowCurve2.jpg',
   'whiteCarLaneSwitch.jpg',
]
outdir = 'test_images_output/'

saveimage = True

def convert(fname):
    image = mpimg.imread('test_images/' + fname)

    kernel_size = 5
    low_threshold = 50
    high_threshold = 150
    imshape = image.shape
    ysize = imshape[0]
    xsize = imshape[1]
    points = [(0,ysize), (xsize*0.46, ysize*0.6), (xsize*0.54, ysize*0.6), (xsize,ysize)]
    vertices = np.array([points], dtype=np.int32)
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 40 # minimum number of pixels making up a line
    max_line_gap = 20    # maximum gap in pixels between connectable line segments

    _glb = Global()

    gray = grayscale(image)
    if saveimage:
        plt.imshow(gray)
        plt.savefig(outdir+'gray_'+fname)

    blur_gray = gaussian_blur(gray, kernel_size)
    if saveimage:
        plt.imshow(blur_gray)
        plt.savefig(outdir+'blur_gray_'+fname)

    edges = canny(blur_gray, low_threshold, high_threshold)
    if saveimage:
        plt.imshow(edges)
        plt.savefig(outdir+'edges_'+fname)

    masked_edges = region_of_interest(edges, vertices)
    if saveimage:
        plt.imshow(masked_edges)
        plt.savefig(outdir+'masked_edges_'+fname)

    line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    if saveimage:
        plt.imshow(line_img)
        plt.savefig(outdir+'line_img_'+fname)

    line_edges = weighted_img(image, line_img)
    if saveimage:
        plt.imshow(line_edges)
        plt.savefig(outdir+'line_edges_'+fname)

    # plot vertices
    #x = [a[0] for a in points]
    #y = [a[1] for a in points]
    #plt.plot(x, y, 'b--', lw=2)

# convert(fnames[0])
for fname in fnames:
    convert(fname)
