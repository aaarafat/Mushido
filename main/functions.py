from commonfunctions import *


def segement_hand_written(img):
    h, w = img.shape
    window, width = get_dimensions(w, h)
    fimg = median(img)
    glares = (fimg > 195)
    bimg = binraization(fimg, window, 7)
    output = (bimg) - (glares)
    output = median(output)
    output = binary_dilation(1-output)
    vse = np.array([[0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0]])

    hse = np.array([[0, 0, 0],
                    [1, 1, 1],
                    [0, 0, 0]
                    ])
    output = binary_dilation(output, vse)
    output = binary_dilation(output, hse)
    output = 1 - output
    return output  # the reuslt is binary image


def separateStaffs(img, thickness, distance):
    rows_sum = np.sum(1 - img, axis=1)
    lines, _ = find_peaks(
        rows_sum, height=img.shape[1] * 0.5, distance=distance+thickness//2)
    parts = [0]

    for i in range(len(lines) - 1):
        if lines[i + 1] - lines[i] > 1.5 * distance:
            parts.append((lines[i + 1] + lines[i]) // 2)

    parts.append(img.shape[0]-1)

    return parts


def getStaffs(image):
    h, w = image.shape
    window, width = get_dimensions(w, h)
    hist = histogram(image, nbins=256, source_range='dtype')
    hist_acc = np.cumsum(hist[0])
    tratio = ((hist_acc[255]-hist_acc[206] +
               hist_acc[49]-hist_acc[0]) / hist_acc[-1])*100
    segmented = []
    if tratio < 4:
        segmented = segement_hand_written(image)
    elif 75 > tratio > 4:
        segmented = binraization(image, window, 35)
    else:
        segmented = (image > 150).astype("uint8")

    thickness, distance = get_lengthes(segmented)
    fimg, candidates = get_candidates_lines(segmented, thickness)
    fimg = filter_candidates_lines(fimg, candidates, thickness, distance)

    parts = separateStaffs(fimg, thickness, distance)
    staffs = []
    for start, end in zip(parts[:-1], parts[1:]):
        staffs.append(segmented[start:end, :])
    return staffs


target_img_size = (32, 32)


def extract_hog_features(img):
    """
    TODO
    You won't implement anything in this function. You just need to understand it 
    and understand its parameters (i.e win_size, cell_size, ... etc)
    """
    img = cv2.resize(img, target_img_size)
    win_size = (32, 32)
    cell_size = (4, 4)
    block_size_in_cells = (2, 2)

    block_size = (block_size_in_cells[1] * cell_size[1],
                  block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 9  # Number of orientation bins
    hog = cv2.HOGDescriptor(win_size, block_size,
                            block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten()
    return h.flatten()


def getDots(img_isolated, t, d):
    contours, _ = cv2.findContours(
        img_isolated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if (d//4 < w < d//2 and d//4 < h < d//2):
            deltax = w//4
            deltay = h//4
            sum = np.sum(img_isolated[y+deltay:y+h -
                                      deltay, x+deltax:x+w-deltax]//255)
            if sum/((h-2*deltay)*(w-2*deltax)) > 0.9:
                dots.append((x, y, w, h))
    return dots


def getContours(img, t, d):
    kernel = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ], np.uint8)
    img = cv2.dilate(img, kernel, iterations=10)  # to merge 2 number together
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    cdots = []
    dots = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > d//2 and h > d:
            i = len(boxes) - 1
            # remove overlaped contours
            if i == -1 or not (x + w <= boxes[i][0] + boxes[i][2] and x >= boxes[i][0]):
                sum = np.sum((img[y:y+h, x:x+w]//255))
                if(sum/(h*w) < .95):
                    boxes.append((x, y, w, h))
    return np.array(boxes)


def xProjection(img, r):
    "Return a list containing the sum of the pixels in each column"
    (h, w) = img.shape[:2]
    sumCols = []
    for j in range(0, w, r):
        mx = 0
        tmp = 0
        for i in range(h):
            if np.all(img[i, j: j + r] == 255):
                tmp += 1
            else:
                mx = max(mx, tmp)
                tmp = 0
        mx = max(mx, tmp)
        sumCols.append(mx)
    return np.array(sumCols)


def yProjection(img, r):
    "Return a list containing the sum of the pixels in each row"
    (h, w) = img.shape[:2]
    sumRows = []
    for i in range(0, h, r):
        tmp = 0
        for j in range(w):
            if img[i: i + r, j] == 255:
                tmp += 1
        sumRows.append(tmp)
    return np.array(sumRows)


def getBeamHeads(img, d):
    x = xProjection(img, 8)
    x[x < d // 2] = 0

    points = []
    tx1, tx2 = -1, -1
    for i in range(len(x)):
        if x[i]:
            if tx1 == -1:
                tx1 = i * 8

        elif tx1 != -1:
            tx2 = i * 8
            points.append((tx1, tx2))
            tx1, tx2 = -1, -1

    if tx1 != -1:
        tx2 = (len(x) - 1) * 8
        points.append((tx1, tx2))

    mask = img.copy()
    mask[:] = 0

    (h, w) = img.shape[:2]
    for px1, px2 in points:
        cv2.rectangle(mask, (px1, 0), (px2, h), (255, 255, 255), -1)

    img[mask != 255] = 0
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=3)

    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h > d // 2:
            i = len(boxes) - 1
            # remove overlaped contours
            if i == -1 or not (x + w <= boxes[i][0] + boxes[i][2] and x >= boxes[i][0]):
                boxes.append((x, y, w, h))

    return np.array(boxes)


def getNoteHeads(img, d):
    y = yProjection(img, 1)
    th = np.max(y) - 5
    y[y < th] = 0

    points = []
    ty1, ty2 = -1, -1
    for i in range(len(y)):
        if y[i]:
            if ty1 == -1:
                ty1 = i

        elif ty1 != -1:
            ty2 = i
            points.append((ty1, ty2))
            ty1, ty2 = -1, -1

    if ty1 != -1:
        ty2 = (len(y) - 1)
        points.append((ty1, ty2))

    boxes = []
    (h, w) = img.shape[:2]

    if len(points):
        points = sorted(points)
        y1, y2 = points[0]
        boxes.append((0, y1, w, y2-y1))

    return np.array(boxes)


def getChordHeads(img, d):
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.erode(img, kernel, iterations=3)
    y = yProjection(img, 1)
    y[y < d//2] = 0

    points = []
    ty1, ty2 = -1, -1
    for i in range(len(y)):
        if y[i]:
            if ty1 == -1:
                ty1 = i

        elif ty1 != -1:
            ty2 = i
            points.append((ty1, ty2))
            ty1, ty2 = -1, -1

    if ty1 != -1:
        ty2 = (len(y) - 1)
        points.append((ty1, ty2))

    boxes = []
    (h, w) = img.shape[:2]

    if len(points):
        points = sorted(points)
        for p in points:
            y1, y2 = p
            boxes.append((0, y1, w, y2-y1))

    return np.array(boxes)


def getLines(img):
    img *= 255
    img = 255 - img
    y = yProjection(img, 1)
    y[y < 100] = 0

    points = []
    ty1, ty2 = -1, -1
    for i in range(len(y)):
        if y[i]:
            if ty1 == -1:
                ty1 = i

        elif ty1 != -1:
            ty2 = i
            points.append((ty1, ty2))
            ty1, ty2 = -1, -1

    if ty1 != -1:
        ty2 = (len(y) - 1)
        points.append((ty1, ty2))

    if not points:
        return np.array(points), 0

    points = np.array(points)
    points = [(p[0] + p[1]) // 2 for p in points]

    d = 0
    for i in range(1, len(points)):
        d += abs(points[i - 1] - points[i])
    d //= len(points) - 1

    p1 = points[0]
    for i in range(1):
        tmp = p1 - (i + 1) * d
        points.insert(0, tmp)
    p1 = points[-1]
    for i in range(1):
        tmp = p1 + (i + 1) * d
        points.append(tmp)

    tmp_points = []

    for p in points:
        tmp_points.append(p)
        tmp_points.append(p + (d // 2))

    tmp_points.pop(len(tmp_points) - 1)
    p1 = tmp_points[0] - (d // 2)
    tmp_points.insert(0, p1)

    return tmp_points, d


def notHalf(img, d):
    x = xProjection(img, 1)
    x[x < d//2] = 0
    x[x > 2*d] = 0
    peak = np.max(x)
    return peak >= d
