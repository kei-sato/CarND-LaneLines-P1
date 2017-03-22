**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Test the pipeline on test images
* Test the pipeline on test videos
* Reflect on my work in a written report

[//]: # (Image References)

[image1]: ./images/solidYellowLeft-1.gif "solidYellowLeft"
[image2]: ./images/challenge-1.gif "challenge"

---

### Reflection

### 1. Description of my pipeline.

My pipeline consisted of 5 steps as follows

* Convert the image to grayscale
* Apply Gaussian smoothing
* Apply Canny edge detection
* Apply polygon mask
* Apply Hough algorithm on edge detected image
* Apply extrapolating lines to draw single lines on the left and right lanes

In order to draw a single line on the left and right lanes, I modified the draw_lines() function as follows

* Calculating slopes of all lines provided
* Remove lines with nearly flat slopes
* Split lines into two groups which either have positive slopes or negative slopes
* Remove lines which have outlier slopes (too far from medium slope value of the group)
* Exrapolating slopes and intercepts by fitting to the points of the lines of the group using LinearRegression
* Add a certain amount of slopes and intercepts to the previous slopes and intercepts (concepts of momentum)
* Extend lines from the bottom of the image to the top of the all lines of the group

![solidYellowLeft][image1]

### 2. Potential shortcomings with the current pipeline

The current pipeline's shortcomings could be as follows

* The region of interest is insensitive to the direction of the car. So it could be useless when the car turns right or left.
* The detection of lanes becomes less reasonable when obstacles or shadows come in.
* The pipeline doesn't seem to work as well with more curved lane lines

![challenge][image2]

### 3. Possible improvements

A possible improvement would be to implement more intelligent way to detect lane lines and the region of interest.
