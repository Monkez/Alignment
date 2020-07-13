import cv2
import numpy as np
from scipy import ndimage
from PIL import Image

import numpy as np
ZERO_IMAGE =  np.empty((0,0,3))

SHOW_IMAGE = True
def show(np_img, show=SHOW_IMAGE):
    if show:
        Image.fromarray(np_img).show()

NOISE_HEIGHT = 10 # 11 is ok for rk
NOISE_WIDTH = 4 # 

def bradley_binarization(image, threshold=90, window_r=55, is_gray=False, inversed=False):
    """
    form5: 90
    form3: 80
    form3: 70
    form2: 90
    form1: ?
    
    """
    percentage = threshold / 100.
    window_diam = 2*window_r + 1
    # convert image to numpy array of grayscale values
    img = np.copy(image)
    if not is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # matrix of local means with scipy
    means = ndimage.uniform_filter(img, window_diam)
    # result: 0 for entry less than percentage*mean, 255 otherwise 
    height, width = img.shape[:2]
    result = np.zeros((height,width), np.uint8)   # initially all 0
    result[img >= percentage * means] = 255       # numpy magic :)
    # Get inversed image if inversed=True
    if inversed:
        return 255 - result
    return result

def find_top_line_from_center(img, lineness, offset):
    height = img.shape[0]
    width = img.shape[1]
    projected_value = np.sum(img, axis=1)/width
    y = height//2
    reach_line = False
    while not reach_line and y > 0:
        y -= 1
        if projected_value[y] > lineness * 255:
            reach_line = True
    # Only aplly offset for case line boundary detected
    if reach_line:
        return y + offset 
    else:
        return y

def find_bottom_line_from_center(img, lineness, offset):
    height = img.shape[0]
    width = img.shape[1]
    projected_value = np.sum(img, axis=1)/width
    y = height//2
    reach_line = False
    while not reach_line and y < height-1:
        y += 1
        if projected_value[y] > lineness * 255:
            reach_line = True
    # Only aplly offset for case line boundary detected
    if reach_line:
        return y - offset 
    else:
        return y

def find_left_line_from_center(img, lineness, offset):
    height = img.shape[0]
    width = img.shape[1]
    projected_value = np.sum(img, axis=0)/height
    y = width//2
    reach_line = False
    while not reach_line and y > 0:
        y -= 1
        if projected_value[y] > lineness * 255:
            reach_line = True
    # Only aplly offset for case line boundary detected
    if reach_line:
        return y + offset 
    else:
        return y

def find_right_line_from_center(img, lineness, offset):
    height = img.shape[0]
    width = img.shape[1]
    projected_value = np.sum(img, axis=0)/height
    y = width//2
    reach_line = False
    while not reach_line and y < width-1:
        y += 1
        if projected_value[y] > lineness * 255:
            reach_line = True
    # Only aplly offset for case line boundary detected
    if reach_line:
        return y - offset 
    else:
        return y

def remove_boundary_offset(img, offset):
    if offset != 0:
        return img[offset:-offset, offset:-offset]
    else: 
        return img

def find_center_text_around_by_lines(aligned_img, is_binarized=False, lineness=0.62, offset = 2, binary_threshold=90):
    # Find lines, text in white, background in black
    img = np.copy(aligned_img)
    # show(img)
    if not is_binarized:
        img = bradley_binarization(aligned_img, threshold=binary_threshold, inversed=True)
    # show(img)
    top = find_top_line_from_center(img, lineness, offset)
    bottom = find_bottom_line_from_center(img, lineness, offset)
    result = img[top:bottom,:]
    # show(img)
    origin_result = aligned_img[top:bottom,:]
    if top < bottom:
        left = find_left_line_from_center(result, 0.85, offset) # always should use hard lineness for left and right
        right = find_right_line_from_center(result, 0.85, offset)
        # print("{}-{}-{}-{}".format(top, bottom, left, right))
        result = result[:, left:right]
        # show(result)
        origin_result = origin_result[:, left:right]
    # result = remove_boundary_offset(result, offset)
    # origin_result = remove_boundary_offset(origin_result, offset)
    return result, origin_result

def is_noise(cc_stat, min_height=NOISE_HEIGHT, min_width=NOISE_WIDTH, min_area=10, noise_wh_ratio=15):
    height = cc_stat[3] # height
    width = cc_stat[2]
    # print(width)
    # print(NOISE_WIDTH)
    area = cc_stat[4]
    return height<=min_height or area<=min_area or width<=min_width or (width/height) > noise_wh_ratio

def is_boundary():
    """
    TODO: touch boundary and length is great enough (for stick to letter case e.g 3_IMG_2883)
    """
    pass

def connected_components_bbox(binarized_img, origin_img, cc_stats):
    # show(origin_img)
    # check why origin disappear
    H, W = binarized_img.shape
    x2 = y2 = 0
    x1 = W
    y1 = H
#     print("H/7 = {}".format(H/7))
    for stat in cc_stats[1:]:
        left = stat[0]
        top = stat[1]
        right = stat[0] + stat[2]
        bottom = stat[1] + stat[3]
        if not is_noise(stat):
            x1 = left if (x1>left) else x1
            y1 = top if (y1>top) else y1
            x2 = right if (x2<right) else x2
            y2 = bottom if (y2<bottom) else y2
        else:
            # Remove noise in, ONLY on binarized_img
            binarized_img[top:bottom, left:right] = 0
    if x2>x1 and y2>y1:
        return binarized_img[y1:y2, x1:x2], origin_img[y1:y2, x1:x2]
    else:
        return ZERO_IMAGE, ZERO_IMAGE #binarized_img, origin_img # return zero image!!!!
    
def centralize_text(binarized_img, origin_img, auto_padding=True):
    """
    binarized_img has text in white, background in black
    """
    connectivity = 4
    # Perform the operation
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized_img, connectivity, cv2.CV_32S)
    tight_text, origin_tight_text = connected_components_bbox(binarized_img, origin_img, stats)
    h = tight_text.shape[0]
    centralized_text = np.pad(tight_text, (h,h), 'constant', constant_values=(0, 0))
    if auto_padding:
        median_value = np.percentile(origin_img, 75, axis=(0, 1))
        origin_centralized_text = np.ones(np.array(origin_tight_text.shape) + (h*2,h*2,0)) * median_value
        origin_centralized_text[h:origin_tight_text.shape[0]+h, h:origin_tight_text.shape[1]+h,:] = origin_tight_text
        return 255 - centralized_text, origin_centralized_text.astype(np.uint8) # return binarized text in black, background in white

    return 255 - centralized_text, origin_tight_text

def countCC1D(binarized_arr, arr):
    """
    """
    cc_list = np.zeros(len(arr), dtype=np.int)
    cc_area = np.zeros(len(arr))
    nb_cc = 0
    if binarized_arr[0] > 0:
        cc_list[0] = 1
        nb_cc += 1
        cc_area[0] += arr[0]
    for i in range(1,len(arr)):
        if binarized_arr[i] > 0:
            if cc_list[i-1] != 0:
                cc_list[i] = cc_list[i-1]
                cc_area[nb_cc-1] += arr[i]
            else:
                cc_list[i] = nb_cc + 1
                nb_cc += 1
                cc_area[nb_cc-1] = arr[i]           
    return cc_list, nb_cc, cc_area[:nb_cc]

def region_avarage_bradley_threshold_1D(array, region_r=1, threshold=50, window_r=30):
    """
    Special Bradley binarization implementation. Relately high-value region-wise => 255.
    """
    percentage = threshold / 100.
    window_diam = 2*window_r + 1
    region_diam = 2*region_r + 1
    arr = np.array(array).astype(np.float)
    # matrix of global means with scipy
    global_means = ndimage.uniform_filter(arr, window_diam)
    # matrix of region means with scipy
    region_means = ndimage.uniform_filter(arr, region_diam)
    # result: 0 for entry less than percentage*global_means, 255 otherwise 
    length = len(arr)
    result = np.zeros((length), np.uint8)   # initially all 0
    result[region_means > percentage * global_means + 0.001] = 255       # numpy magic :)
    return result

def get_largest_binarized_CC1D(binarized_arr, arr, nb_returns=1):
    cc_list, nb_cc, cc_area = countCC1D(binarized_arr, arr)
    top_index = np.flip(np.argsort(cc_area), axis=0)[:min(nb_cc, nb_returns)]
    sorted_index = np.sort(top_index)
    return [(max(1,np.where(cc_list==(i+1))[0][0]), min(np.where(cc_list==(i+1))[0][-1], len(arr)-1)) for i in sorted_index]


def get_CC_below_top(binarized_arr, arr, offset=1, noise_size=NOISE_HEIGHT):
    # print(arr.shape)
    H = len(binarized_arr)
    cc_list, nb_cc, cc_area = countCC1D(binarized_arr, arr)
    # print(cc_list)
    # print(nb_cc)
    # print(cc_area)
    if nb_cc > 0:
        for i in range(1, nb_cc+1):
            idx = np.where(cc_list == i)[0]
            if idx[-1] - idx[0] > noise_size:
                return (max(idx[0] - offset, 0), min(idx[-1] + offset, H-1))
        # return empty image
        return (0, 0)
    else:
        return (0, H-1)

def cut_top_line(text_img, origin_text_img):
    # Input: Gray image. Text in black on white background
    w = text_img.shape[1]
    projected_value = np.sum(text_img, axis=1)/w
    binarized_his_on_y = region_avarage_bradley_threshold_1D(projected_value)
    # result = get_CC_below_top(binarized_his_on_y, projected_value)
    result = get_largest_binarized_CC1D(binarized_his_on_y, projected_value, nb_returns=1)
    if not result:
        result = (0,0) # return zero img
    else:
        result = result[0]
    return text_img[result[0]:result[1], :], origin_text_img[result[0]:result[1], :]
    
def text_value_regressor(img, binary_threshold=90, auto_padding=True):
    """
    Input: original crop from deskewed payslip in RGB.
    """
    text_img, origin_text_img = find_center_text_around_by_lines(img, is_binarized=False, binary_threshold=binary_threshold)
    if origin_text_img.size:
        bottom_text_line, origin_bottom_text_line = cut_top_line(text_img, origin_text_img)
        if origin_bottom_text_line.size:
            # print(origin_bottom_text_line.size)
            output, origin_output = centralize_text(bottom_text_line, origin_bottom_text_line, auto_padding=auto_padding)
            return output, origin_output
        else:
            return bottom_text_line, origin_bottom_text_line
    else:
        return text_img, origin_text_img
    
