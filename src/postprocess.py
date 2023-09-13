import numpy as np
import time

def non_max_suppression(prediction, conf_thres, iou_thres, merge=False, classes=None, multi_label=False, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    #Changed to adapt to 84 anchor boxes.
    nc = prediction.shape[1] - 4  # number of classes
    #generate our boolean array and get confidence values from one specific anchor box
    xc = np.max(prediction[:, 4:84], axis=1) > conf_thres #candidates

    #Changed the max_wh, max_det to heigher values
    max_wh = 7680  # (pixels) minimum and maximum box width and height
    max_det = 30000  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        #change the tensor from 84, 8400 -> 8400 84 and then with (num of T values, 84)
        x = x.transpose()[xc[xi]] 
       

        # If none remain process next image
        if not x.shape[0]:
            continue

        #extract the bboxes, cls
        nm = 0 #number of masks
        box = np.split(x, [4], axis=1)[0]
        cls = np.split(x, [4, 4+nc], axis=1)[1]
        mask = np.split(x, [4+nc, 4+nc+nm], axis=1)[2]
        

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(box)

        #Unpacking to get classes 
        i, j = np.nonzero(cls > conf_thres)
        
        #Detections matrix nx6 (xyxy, conf, cls), 
        x = np.concatenate((box[i], x[i, j + 4, None], j[:, None].astype('float16')), 1)

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes, scores, iou_thres)
       
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
            
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

#bounding box coordinates computation, scaling and others remained the same, they are pretty standard
#edited from float32 to float16 to match our feature map data.
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    #y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    
    return y


def nms(boxes, scores, iou_thres):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return np.zeros(1)
    
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float16")
    
    # initialize the list of picked indexes	
    pick = []
    
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)
    
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > iou_thres)[0])))
        
    # return only picked value
    return np.array(pick)


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img_shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img_shape[0])  # y1, y2