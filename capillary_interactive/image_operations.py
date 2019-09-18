from scipy.ndimage import uniform_filter
from scipy.ndimage import gaussian_filter

import zipfile
import os

from PIL import Image
import shutil

import numpy as np
import matplotlib.pyplot as plt
import cv2


from matplotlib import gridspec
import matplotlib.pyplot as plt

def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

def color_plot(images,figsize=(12, 8),dpi=200):
    """fig = plt.figure(figsize=figsize,dpi=dpi)
    fig.subplots_adjust(wspace=0, hspace=0)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, .5])
    ax = [plt.subplot(gs[x]) for x in range(2)]

    for img in images:
        is_purple, hist_hue_purple = check_purple(img)
        ax[0].plot(hist_hue_purple)
    ax[0].set_xlim(0,99)
    ax[1].imshow([np.linspace(0, 100, 100)], aspect='auto', cmap=plt.get_cmap("hsv"))
    ax[0].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[1].set_xticks([x for x in range(0,99,5)])
    return fig2rgb_array(fig)"""
    pass

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

def crop(image_array,top,left,bottom,right):
    image_array = image_array.copy()
    return image_array[top:bottom,left:right]

def blur_columnwise(image_array, repetitions = 30, size=10):
    """ Blurs columnwise: Smoothing along dimension y.
    Args:
        image_array: The image to be processed. A 2d Numpy array.
        repetitions: Defines how many time uniform filtering is applied
        size: The size of the uniform filter kernel.
    Returns:
        The filtered image.
    """
    image_array = image_array.copy()
    for repetition in range(repetitions):
        for x in range(image_array.shape[1]):#for each column: filter seperately
            image_array[:,x] = uniform_filter(image_array[:,x],10)

    return image_array

def blur(image_array, repetitions = 10, size = 10):
    """ Blurs image using uniform filtering.
    Args:
        image_array: The image to be processed. A 2d Numpy array.
        repetitions: Defines how many time uniform filtering is applied
        size: The size of the uniform filter kernel.
    Returns:
        The filtered image.
    """
    image_array = image_array.copy()

    for x in range(repetitions):
        image_array = uniform_filter(image_array,10)
    return image_array

def gradient_columnwise(image_array):
    """ Computes the gradient for each column.
    Args:
        image_array: The image to be processed. A 2d Numpy array.
    Returns:
        The filtered image.
    """
    image_array = image_array.copy()
    for x in range(image_array.shape[1]):
        image_array[:,x] = np.gradient(image_array[:,x])
    return image_array

def gaussian_filter_nan(array, sigma):
    U = np.asarray(array)

    if np.isnan(U[0]):
        U[0]=np.nanmean(U)
    if np.isnan(U[-1]):
        U[-1]=np.nanmean(U)

    V=U.copy()
    V[np.isnan(U)]=0
    VV=gaussian_filter(V,sigma=sigma)

    W=0*U.copy()+1
    W[np.isnan(U)]=0
    WW= gaussian_filter(W,sigma=sigma)

    epsilon=0.000001
    Z=VV/(WW+epsilon)
    return Z

def flip_ud(y,image_height):
    return -y+ image_height

def sample_via_threshold(image_array,start_from="top", threshold=.6, start = 500, shift = 0):
    """ Columnwise finds the first point where the image_array exceeds the threshold starting at start_from."""
    ran = None
    if start_from == "top":
        ran = range(start,image_array.shape[0],1)#From top to bottom
    elif start_from == "bottom":
        ran = range(start,0,-1)#From bottom to top

    sampled_points = []


    for x in range(image_array.shape[1]):
        for y in ran:#From bottom to top
            found = False
            if(np.abs(image_array[y,x])>threshold):
                found = True
                sampled_points.append(y+ shift)
                break
        if not found:
            sampled_points.append(np.nan)

    return sampled_points

def horizontal_component(image_array):
    """ Retrieves horizontal component of gradient and standardizes the resulting values using the standard deviation"""
    horizontal_component = np.asarray(np.gradient(gaussian_filter(image_array,1))[0],dtype=np.float)
    horizontal_component = horizontal_component+np.abs(np.min(horizontal_component))
    horizontal_component /= np.std(horizontal_component)
    return horizontal_component


def remove_isolated_pixels(image_array, kernel_y=4, kernel_x=6):
    """ Processes a binary image and removes isolated pixel groups using the binary opening operator
    Args:
        image_array: 2D Numpy array containing binary values.
        kernel_y: Size of kernel for opening in dimension y
        kernel_x: Size of kernel for opening in dimension x
    """
    image_array = np.asarray(image_array,dtype=np.float)
    kernel = np.ones((kernel_y,kernel_x),np.uint8)
    opening = cv2.morphologyEx(image_array, cv2.MORPH_OPEN, kernel)
    return opening

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

function_to_default_params = {}
function_to_default_params["crop"] = [200,20,-200,-20]
function_to_default_params["blur columnwise"]=[30,10]
function_to_default_params["blur"] = [20,10]
function_to_default_params["gradient columnwise"] = []
function_to_default_params["sample via threshold"]=["top",0.6,200,0]
function_to_default_params["gaussian filter nan"]=[4]
function_to_default_params["flip_ud"] = [1000]
function_to_default_params["horizontal component"] = []
function_to_default_params["remove isolated pixels"] = [4,6]
