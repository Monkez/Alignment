import numpy as np
import cv2
from transform import four_point_transform
from sklearn.cluster import KMeans
from regressor import show
import imutils
HEIGHT = 512
WIDTH = 512


def intersection(line1, line2):
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return np.array([x0, y0])


def draw_line(img, line, color):
    r, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * r
    y0 = b * r
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(img, (x1, y1), (x2, y2), color, 2)


def get_rect_by_lines(img, model, name):
    img2 = img.copy()
    color = [(0, 255, 0),
                 (0, 0, 255),
                 (255, 0, 0),
                 (255, 255, 0)]
    results = None
    min_lenght = 200
    edge = model.predict(np.array([img/255.0]))[0]
    edge = (255*(edge+1)/2).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    edge = cv2.erode(edge, kernel, iterations=1)
    img_show = img.copy()
    ret, edge = cv2.threshold(edge,30,255,cv2.THRESH_BINARY)
    img_show = np.hstack((img_show, cv2.cvtColor(edge,cv2.COLOR_GRAY2BGR)))
    lines = cv2.HoughLines(edge, 1, np.pi/360, 150)
    if lines is None or len(lines)<4:
        return None
    lines = [line[0] for line in lines]
    lines = np.array(lines)
    lines2 = lines.copy()+0
    for i in range(len(lines2)):
        lines2[i][0] = abs(lines2[i][0])
        if lines2[i][1]>np.pi/2:
            lines2[i][1] = np.pi - lines2[i][1]
        lines2[i][1] = 100*(lines2[i][1])
    kmeans = KMeans(n_clusters=4, random_state=0).fit(lines2)
    pred_label = kmeans.predict(lines2)
    for i in range(len(lines)):
        line = lines[i]
        draw_line(img2, line, color[pred_label[i]])

    strongest_lines = []
    this_lines = lines[pred_label==1]
    for i in range(4):
        this_lines = lines[pred_label==i]
        if ( np.max(this_lines[:, 1]) - np.min(this_lines[:, 1]))>2:
            for j in range(len(this_lines)):
                if this_lines[j][1]>np.pi-0.3:
                    this_lines[j][1] = this_lines[j][1]-np.pi
                    this_lines[j][0] = - this_lines[j][0]
                
        mean_line = np.mean(this_lines, axis=0)
        if mean_line[1]<0:
            mean_line[1]+=np.pi
            mean_line[0] = - mean_line[0]
        strongest_lines.append(mean_line)
        draw_line(img, mean_line, color[i])
        
    strongest_lines = np.array(strongest_lines)
    theta = strongest_lines[:, 1].copy()
    theta[theta>np.pi/2] = np.pi-theta[theta>np.pi/2]
    theta = theta.reshape((4, 1))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(theta)
    classes = kmeans.predict(theta)
    lines1 = strongest_lines[classes==0]
    lines2 = strongest_lines[classes==1]
    img_show = np.vstack((img_show,  np.hstack((img2, img))))
    cv2.imwrite("out/line_process.jpg", img_show)
    if lines1.shape[0]!=2:
        return None	
    point1 = intersection(lines1[0], lines2[0])
    point2 = intersection(lines2[0], lines1[1])
    point3 = intersection(lines1[1], lines2[1])
    point4 = intersection(lines2[1], lines1[0])
    return np.array([point4, point1, point2, point3])


def adjust_image(img):
    orin = img.copy()
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mean = np.mean(gray) - 20
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 21)
    _mean = np.mean(thresh[0])
    stt = 0
    max_px = 0.1
    while (_mean < 30) and stt < max_px*h:
        stt+=1
        thresh = thresh[1:]
        img = img[1:]
        _mean = np.mean(thresh[0])

    _mean = np.mean(thresh[-1])
    stt = 0
    while (_mean < 30) and stt < max_px*h:
        stt+=1
        thresh = thresh[:-1]
        img = img[:-1]
        _mean = np.mean(thresh[-1])

    _mean = np.mean(thresh[:, 0])
    stt = 0
    while (_mean < 30) and stt < max_px*w:
        stt+=1
        thresh = thresh[:, 1:]
        img = img[:, 1:]
        _mean = np.mean(thresh[:, 0])

    _mean = np.mean(thresh[:, -1])
    stt = 0
    while (_mean < 30)  and stt < max_px*w:
        stt+=1
        thresh = thresh[:, :-1]
        img = img[:, :-1]
        _mean = np.mean(thresh[:, -1])
    return img


def four_point_transform_with_mask(mask, orin, box):
    warped_mask = four_point_transform(mask, box)
    mask_shape = mask.shape
    orin_shape = orin.shape
    w, h = mask_shape[0:2]
    W, H = orin_shape[0:2]
    rh = H / h
    rw = W / w
    BOX = np.zeros_like(box)
    BOX[:, 0] = box[:, 0] * rh
    BOX[:, 1] = box[:, 1] * rw
    BOX = BOX.astype(np.int16)
    warped_orin = four_point_transform(orin, BOX)
    return warped_mask, warped_orin


def get_rect(img, min_area):
    img = cv2.Canny(img, 70, 200)
    w, h = img.shape[0:2]
    t = 5
    img = cv2.resize(img,(h+2*t, w+2*t))
    img = img[t:-t, t:-t]
    h, w = img.shape
    screenCnt = None
    cnts = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        if len(approx) == 4 and cv2.contourArea(c) > h * w * min_area:
            screenCnt = approx
            break
    if screenCnt is not None:
        box = screenCnt.reshape(4, 2)
        return box
    else:
        return None


def get_bounding(mask, orin_image):
    small_size = 800
    ratio = mask.shape[0] / small_size
    box = None
    kernel = np.ones((5, 5), np.uint8)
    orin_mask = mask.copy()
    mask = cv2.resize(mask, (int(mask.shape[1] / ratio), small_size), interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(mask, 100, 200)
    box = get_rect(edges, 0.3)
    if box is None:
        ret, edge = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)
        gray = cv2.GaussianBlur(edge, (5, 5), 0)
        threshold_param = 200
        while box is None and threshold_param > 100:
            ret, Threshold = cv2.threshold(gray, threshold_param, 255, cv2.THRESH_BINARY)
            box = get_rect(Threshold, 0.3)
            threshold_param -= 5
            if box is not None:
                Threshold = cv2.cvtColor(Threshold, cv2.COLOR_GRAY2BGR)
                cv2.line(Threshold, tuple(box[0]), tuple(box[1]), (255, 0, 0), 2)
                cv2.line(Threshold, tuple(box[1]), tuple(box[2]), (255, 0, 0), 2)
                cv2.line(Threshold, tuple(box[2]), tuple(box[3]), (255, 0, 0), 2)
                cv2.line(Threshold, tuple(box[3]), tuple(box[0]), (255, 0, 0), 2)
                cv2.imwrite("out/Threshold.jpg", Threshold)
    if box is not None:
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.line(edges, tuple(box[0]), tuple(box[1]), (255, 0, 0), 2)
        cv2.line(edges, tuple(box[1]), tuple(box[2]), (255, 0, 0), 2)
        cv2.line(edges, tuple(box[2]), tuple(box[3]), (255, 0, 0), 2)
        cv2.line(edges, tuple(box[3]), tuple(box[0]), (255, 0, 0), 2)
        cv2.imwrite("out/candy_edge.jpg", edges)
        box = (box * ratio).astype(np.int16)
        warped_mask, warped_orin = four_point_transform_with_mask(orin_mask, orin_image, box)
        t = 0.01
        w, h = warped_mask.shape[0:2]
        warped_mask = warped_mask[int(w*t):w - int(w*t)]
        w, h = warped_orin.shape[0:2]
        warped_orin = warped_orin[int(w * t):w - int(w * t)]
        return warped_mask, warped_orin
    else:
        return None, None


def straighten(mask, orin, factor=5):
    result = np.array([])
    edge = cv2.Canny(mask, 75, 200)
    mask_shape = mask.shape
    orin_shape = orin.shape
    h, w = mask_shape[0:2]
    H, W = orin_shape[0:2]
    height = h // factor
    points = [[(10, 0), (w-10, 0)]]
    for i in range(factor - 1):
        row = (1 + i) * height
        for j in range(int(w / 6)):
            if edge[row, j] > 200:
                l = (j, row)
                break
            l = (0, row)

        for j in reversed(range(w - int(w / 6), w)):
            if edge[row, j] > 200:
                r = (j, row)
                break
            r = (w, row)
        points.append([l, r])
    points.append([(10, h), (w-10, h)])
    for i in range(len(points) - 1):
        tl = points[i][0]
        tr = points[i][1]
        br = points[i + 1][1]
        bl = points[i + 1][0]
        box = np.array([tl, tr, br, bl])
        crop, crop_orin = four_point_transform_with_mask(mask, orin, box)
        crop_shape = crop.shape
        _h, _w = crop_shape[0:2]
        crop = cv2.resize(crop, (w, _h))
        _H, _W, _ = crop_orin.shape
        crop_orin = cv2.resize(crop_orin, (W, _H))
        if i == 0:
            result = crop
            result_orin = crop_orin
        else:
            result = np.concatenate((result, crop))
            result_orin = np.concatenate((result_orin, crop_orin))
    cv2.imwrite("out/straighten_mask.jpg", result)
    #cv2.imwrite("out/straighten_orin.jpg", result_orin)
    return result_orin


def aligment_with_box(orin_image, model, factor=15):
    HEIGHT = 512
    WIDTH = 512
    image = cv2.resize(orin_image, (HEIGHT, WIDTH))
    edge = model.predict(np.array([image / 255.0]))[0]
    edge = (255 * (edge + 1) / 2).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    edge = cv2.erode(edge, kernel, iterations=1)
    cv2.imwrite("out/edge.jpg", edge)
    warped_mask, warped_orin = get_bounding(edge, orin_image)
    if warped_mask is None:
        return None
    cv2.imwrite("out/warped_mask.jpg", warped_mask)
    cv2.imwrite("out/warped_orin.jpg", warped_orin)
    return straighten(warped_mask, warped_orin, factor)


def align_image_dnn(orin_image, model, name = 'image'):
    warped = aligment_with_box(orin_image, model, 20)
    if warped is None:
        image = cv2.resize(orin_image, (HEIGHT, WIDTH))
        box = get_rect_by_lines(image.copy(), model, name)
        if box is not None:
            print("align with lines algorith")
            box = np.array(box)
            box[:, 0]=box[:, 0]*orin_image.shape[1]/WIDTH
            box[:, 1]=box[:, 1]*orin_image.shape[0]/HEIGHT
            box = box.astype(np.int16)
            warped = four_point_transform(orin_image, box)
    else:
        print("align with box algorith")
    if warped is not None and warped.shape[0]*warped.shape[1]>0:
        final_result = adjust_image(warped)
        t = 15
        final_result = final_result[t:-t, t:-t]
        cv2.imwrite("out/aligned.jpg", final_result)
        return final_result
    else :
        print(" this is oiginal documment!")
        return orin_image