from scipy.ndimage import uniform_filter
from scipy.ndimage import gaussian_filter

import zipfile
import os

from PIL import Image
import shutil

import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage import exposure
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib

from collections import defaultdict
from skimage.transform import (hough_line, hough_line_peaks, probabilistic_hough_line)

import ast

from skimage.util.shape import view_as_windows#For sliding window

def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    img = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    #Image.fromarray(img).save("othertest.png")
    return img

def image_to_cascade_data(image):
    data = defaultdict()
    data["image_array"] = image
    data["original"] = image.copy()
    return data

def color_plot(image, plot = None, figsize=(12,8),dpi=200, scatter=True, slope_intercept = None, markersize = 0.4):
    fig, ax = plt.subplots(figsize=figsize,dpi=dpi)
    plt.tight_layout(pad=0.8)
    #ax.set_xticklabels([])#No labels for the ticks at the x axis
    #ax.set_yticklabels([])
    ax.set_yticks([x*50 for x in range(image.shape[1]//50)])
    #ax.set_xticks([x*50 for x in range(image.shape[0]//50)])

    ax.set_xlim((0, image.shape[1]))
    ax.set_ylim((image.shape[0], 0))

    plt.rcParams.update({'font.size': 12})
    ax.imshow(image, aspect="auto")

    if type(plot)!=type(None):
        if scatter:
            ax.scatter(np.arange(len(plot)),plot, s = (plt.rcParams['lines.markersize']**2)*markersize,c = "red")
        else:
            ax.plot(plot, color="red")

    if slope_intercept:
        x = range(image.shape[1])#x values
        for slope, intercept in slope_intercept:
            ax.plot(x, intercept+x*slope, c = 'orange')

    return fig2rgb_array(fig)


def texture_based_segmentation(data,window_size=10, stride =10, mode = "square"):
    img = data["image_array"]
    mean_std_y = None
    mean_std_x = None
    windows = view_as_windows(img,(window_size,window_size),int(stride))

    if mode == "square":
        collapsed = np.reshape(windows, [windows.shape[0],windows.shape[1],window_size*window_size])
        #collapsed -= np.mean(collapsed,axis=2,keepdims=True)
        mean_std_window = np.std(collapsed,axis=2)
        data["image_array"] = mean_std_window
    if mode == "y" or mode == "combined":
        slice = windows[:,:,window_size//2:(window_size//2)+1,:]
        slice = np.reshape(slice,[slice.shape[0],slice.shape[1],window_size])
        #slice -= np.mean(slice,axis=2,keepdims=True)
        mean_std_y = np.std(slice, axis = 2)
        data["image_array"] = mean_std_y
    if mode == "x" or mode == "combined":
        slice = windows[:,:,:,window_size//2:(window_size//2)+1]
        slice = np.reshape(slice,[slice.shape[0],slice.shape[1],window_size])
        #slice -= np.mean(slice,axis=2,keepdims=True)
        mean_std_x = np.std(slice, axis = 2)
        data["image_array"] = mean_std_x
    if mode == "combined":
        img = (mean_std_x+mean_std_y)/2
        data["image_array"] = img
    return data


def load_docx(path):
    """ Loads a docx file and returns all images contained in it
    Args:
        path: Path to a docx file
    Returns:
        List of images in the form of numpy arrays
    """
    images = []
    zip_ref = zipfile.ZipFile(path, 'r')
    zip_ref.extractall("./TMP/")
    outfiles = os.listdir("./TMP/media")
    for f in outfiles:
        i = Image.open("./TMP/media/"+f)
        images.append(np.array(i))
    zip_ref.close()
    shutil.rmtree("./TMP/")
    return list(images)

def extract_pics_from_docx(path, target_path):
    """ Loads a docx file and extracts images to specified target_path
    Args:
        path: Path to a docx file
        target_path: The folder the images will be written to
    """
    os.makedirs(target_path, exist_ok=True)

    images = []
    zip_ref = zipfile.ZipFile(path, 'r')
    zip_ref.extractall("./TMP/")
    outfiles = os.listdir("./TMP/media")
    for f in outfiles:
        shutil.copy("./TMP/media/"+f,target_path)
    zip_ref.close()
    shutil.rmtree("./TMP/")
    zip_ref.close()

def crop(data,top,left,bottom,right):
    image_array = data["image_array"]
    data["image_array"] = image_array[top:bottom,left:right].copy()
    data["cropped_original"] = image_array[top:bottom,left:right]

    return data

def assign_columnwise(data, base_pos, min_columwidth = 5, pixels_per_micrometer=64):
    points = data["1d_array"]#y values for x in range(0,len(points))
    lines = data["houghlines"]

    image_height = None
    image_width = None
    if "cropped_original" in data:
        image_height = data["cropped_original"].shape[0]
        image_width = data["cropped_original"].shape[1]
    elif "original" in data:
        image_height = data["original"].shape[0]
        image_width = data["original"].shape[1]
    else:
        print("could not determin image_height")
        return

    if int(base_pos) < 0:
        base_pos = image_height

    #Find vertical borders of each column at a reasonable position
    eval_y = np.nanmean(points)#evaluate x position for y coordinates
    x_vals = [(eval_y - intercept)/slope for slope, intercept in lines]
    x_vals.sort()

    #Discard small distances...
    min_columwidth = int(min_columwidth)
    dists = [x2-x1 for x1, x2 in zip(x_vals,x_vals[1:])]
    new_x_vals = []
    merge_now = False
    for i, _ in enumerate(dists):
        if merge_now:
            merge_now = False
            continue
        if dists[i] < min_columwidth:
            prev_dist = None
            next_dist = None
            try:
                prev_dist = dists[i-1]
                next_dist = dists[i+1]
            except:
                pass
            if prev_dist and next_dist:
                if prev_dist < next_dist:#smaller column -> there is no border yet
                    continue
                else:#This is in fact the border but there is no border between next and the one after
                    new_x_vals.append(x_vals[i])
                    merge_now = True
            else:
                new_x_vals.append(x_vals[i])
        else:
            new_x_vals.append(x_vals[i])
    x_vals = new_x_vals
    x_vals.insert(0,0)
    x_vals.append(image_width)

    simplified = np.ndarray(len(points))
    simplified.fill(0)
    columnwise = []

    #evaluate columnwise
    for start, end in zip(x_vals, x_vals[1:]):
        start = int(start)
        end = int(end)
        y_vals = points[start:end]
        best_guess = np.nanmedian(y_vals)
        columnwise.append(best_guess)
        simplified[start:end] = best_guess
    data["1d_array"] = simplified

    data["columnwise"] = (image_height-np.array(columnwise)-(image_height-base_pos))/pixels_per_micrometer
    #del(data["houghlines"])

    return data


def detect_vertical_lines(data, max_deviation_from_vertical = 1, row_begin=100, row_end=-100, min_distance =50):
    if "cropped_original" in data:
        image = data["cropped_original"]
    else:
        image = data["original"]

    image = np.gradient(image)[1]#Vertical component of gradient
    image = image[row_begin:row_end,:]

    h, theta, d = hough_line(image)#hough transform
    h[:,:90-max_deviation_from_vertical] = 0 #Mask out areas that relate to non vertical lines
    h[:,90+max_deviation_from_vertical:] = 0

    h = np.log(1 + h) #log scale for better visibility

    fx = []
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, min_distance = min_distance)):
        y0 = (dist                                 ) / np.sin(angle)#y for x = 0
        y1 = (dist - image.shape[1] * np.cos(angle)) / np.sin(angle)#y for x = image.shape[1]
        intercept = y0
        slope = (y1-y0)/float(image.shape[1])
        fx.append([slope, intercept])

    data["houghlines"] = fx
    #ax.plot(x, intercept+x*slope,  '-r')
    return data

def blur_columnwise(data, repetitions = 30, size=10):
    """ Blurs columnwise: Smoothing along dimension y.
    Args:
        image_array: The image to be processed. A 2d Numpy array.
        repetitions: Defines how many time uniform filtering is applied
        size: The size of the uniform filter kernel.
    Returns:
        The filtered image.
    """
    image_array = data["image_array"]
    image_array = image_array.copy()
    for repetition in range(repetitions):
        for x in range(image_array.shape[1]):#for each column: filter seperately
            image_array[:,x] = uniform_filter(image_array[:,x],10)

    data["image_array"] = image_array
    return data

def equalize_histogram(data):
    """
        Histogram normalization
    """
    image_array = data["image_array"]
    image_array = image_array.copy()
    image_array = exposure.equalize_hist(image_array)
    data["image_array"] = image_array
    return data


def blur(data, repetitions = 10, size = 10):
    """ Blurs image using uniform filtering.
    Args:
        image_array: The image to be processed. A 2d Numpy array.
        repetitions: Defines how many time uniform filtering is applied
        size: The size of the uniform filter kernel.
    Returns:
        The filtered image.
    """
    image_array = data["image_array"]
    image_array = image_array.copy()

    for x in range(repetitions):
        image_array = uniform_filter(image_array,10)
    data["image_array"] = image_array
    return data

def gradient_columnwise(data):
    """ Computes the gradient for each column.
    Args:
        data: A
        image_array: The image to be processed. A 2d Numpy array.
    Returns:
        The filtered image.
    """
    image_array = data["image_array"]

    image_array = image_array.copy()
    for x in range(image_array.shape[1]):
        image_array[:,x] = np.gradient(image_array[:,x])
    data["image_array"] = image_array
    return data

def gaussian_filter_nan(data, sigma):
    array = data["1d_array"]
    vector = np.asarray(array)

    if np.isnan(vector[0]):
        vector[0]=np.nanmean(vector)#TODO
    if np.isnan(vector[-1]):
        vector[-1]=np.nanmean(vector)
    nans, x = np.isnan(vector), lambda z: z.nonzero()[0]
    vector[nans]= np.interp(x(nans), x(~nans), vector[~nans])
    vector = gaussian_filter(vector,sigma=sigma)
    data["1d_array"] = np.array(vector)
    return data

def flip_ud(data,image_height):
    y = data["1d_array"]
    data["1d_array"] = -y+ image_height
    return data

def sample_via_threshold(data,start_from="top", threshold=.6, start = 500, shift = 0):
    """ Columnwise finds the first point where the image_array exceeds the threshold starting at start_from."""
    image_array = data["image_array"]
    if start < 0:
        start = image_array.shape[1]+start
    ran = None
    if start_from == "top":
        ran = range(start,image_array.shape[0],1)#From top to bottom
    elif start_from == "bottom":
        ran = range(start,0,-1)#From bottom to top

    sampled_points = []

    for x in range(image_array.shape[1]):
        found = False
        for y in ran:#From bottom to top
            if(np.abs(image_array[y,x])>threshold):
                found = True
                sampled_points.append(y+ shift)
                break
        if not found:
            sampled_points.append(np.nan)
    data["1d_array"] = sampled_points
    return data

def horizontal_component(data):
    """ Retrieves horizontal component of gradient and standardizes the resulting values using the standard deviation"""
    image_array = data["image_array"]
    horizontal_component = np.asarray(np.gradient(gaussian_filter(image_array,1))[0],dtype=np.float)
    horizontal_component = horizontal_component+np.abs(np.min(horizontal_component))
    horizontal_component /= np.std(horizontal_component)
    data["image_array"] = horizontal_component
    return data


def remove_isolated_pixels(data, kernel_y=4, kernel_x=6):
    """ Processes a binary image and removes isolated pixel groups using the binary opening operator
    Args:
        image_array: 2D Numpy array containing binary values.
        kernel_y: Size of kernel for opening in dimension y
        kernel_x: Size of kernel for opening in dimension x
    """
    image_array = data["image_array"]
    image_array = np.asarray(image_array,dtype=np.float)
    kernel = np.ones((kernel_y,kernel_x),np.uint8)
    opening = cv2.morphologyEx(image_array, cv2.MORPH_OPEN, kernel)
    data["image_array"] = opening
    return data

def apply_cascade(data, cascade):
    """ Applies a cascade of image operations or filters one after the other and returns the final result
    Args:
        cascade: A list of lists containg a function and the values for the parameters 1...n for these functions.
                 Each of named function must be defined such that its first formal parameter relates to the data to be processed.
    Returns:
        The processed data.
    """
    current_data = np.asarray(data,dtype = np.float)
    for function, parameters in cascade:
        current_data = name_to_function[function](current_data,*parameters)
    return current_data

def mask_values(data,remove,add):
    image_array = data["image_array"]
    try:
        remove = ast.literal_eval(remove)
        add = ast.literal_eval(add)
    except:
        raise Exception("Invalid list entered for mask values")

    mask = np.ones(image_array.shape,dtype=np.int32)
    for top, left, bottom, right in remove:
        top = max(0,top)#Make sure none of the values is smaller then 0
        left = max(0,left)
        bottom = max(0,bottom)
        right = max(0,right)

        top = min(mask.shape[0]-1,top)#Make sure none of the values is larger the image
        left = min(mask.shape[1]-1,left)
        bottom = min(mask.shape[0]-1,bottom)
        right = min(mask.shape[1]-1,right)

        mask[top:bottom,left:right] = 0#mask out values

    points = data["1d_array"]
    new_points = []
    for y, x in zip(points,np.arange(len(points))):
        try:
            if(mask[int(y),int(x)]):
                new_points.append(y)
            else:
                new_points.append(np.nan)
        except Exception as e:
            print(e)
            new_points.append(np.nan)
    data["1d_array"] = new_points
    return data

name_to_function = {}
name_to_function["blur columnwise"] = blur_columnwise
name_to_function["sample via threshold"] = sample_via_threshold
name_to_function["flip ud"] = flip_ud
name_to_function["gaussian filter nan"] = gaussian_filter_nan
name_to_function["blur"] = blur
name_to_function["gradient columnwise"] = gradient_columnwise
name_to_function["crop"] = crop
name_to_function["horizontal component"] = horizontal_component
name_to_function["remove isolated pixels"] = remove_isolated_pixels
name_to_function["detect vertical lines"] = detect_vertical_lines
name_to_function["equalize histogram"] = equalize_histogram
name_to_function["assign columnwise"] = assign_columnwise
name_to_function["mask values"] = mask_values
name_to_function["texture based segmentation"] = texture_based_segmentation

function_to_default_params = {}
function_to_default_params["crop"] = [200,20,-200,-20]
function_to_default_params["blur columnwise"]=[30,10]
function_to_default_params["blur"] = [20,10]
function_to_default_params["gradient columnwise"] = []
function_to_default_params["sample via threshold"]=["top",0.6,200,0]
function_to_default_params["gaussian filter nan"]=[4]
function_to_default_params["flip ud"] = []
function_to_default_params["horizontal component"] = []
function_to_default_params["remove isolated pixels"] = [4,6]
function_to_default_params["detect vertical lines"] = [1,100,-100,20]
function_to_default_params["equalize histogram"] =[]
function_to_default_params["assign columnwise"] = [-1, 5]
function_to_default_params["mask values"] = ["[]","[]"]
function_to_default_params["texture based segmentation"] = [50,3,"combined"]
