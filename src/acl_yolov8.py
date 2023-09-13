'''
YOLOv8s Object Detection with YOLOv8s

Testing Data:
- YOLOv5 classic dog.jpg
- Some images I got from COCO2017 dataset
- Some random .mp4

Last Updated: 13-Sep-2023
'''
import os, sys
sys.path.append('../acllite')                  #acllite location
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource
import cv2 
import numpy as np

import utils                             #model information
import preprocess
import postprocess

OM_PATH = "../models/yolov8s.om"
YOLOV8_INPUT_SIZE = (640, 640)
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.6

#TODO: logging
#TODO: threading, or performance improvement, checkout mxbase
#TODO: RTSP camera


def object_detect(path, show_debug=False):
    acl_resource = AclLiteResource()
    acl_resource.init()
    
    device = 0
    yolov8_model = AclLiteModel(OM_PATH, device)
    model_desc = yolov8_model._model_desc
    
    #generate information about the model
    utils.get_sizes(model_desc, show_debug)
    
    if path.endswith(".mp4"):
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)
        print(f"Attempting to open path={path}")
        print(f"Frames/s ={fps}")
        print(f"Frame width ={width}")
        print(f"Frame height ={height}")
        
        #store the inference results
        output_directory = "../result/"
        output_video_path = os.path.join(output_directory, os.path.basename(path).replace('.mp4', '_result.mp4'))
        videoWriter = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, frame_size)

        while cap.isOpened():
            success, img_bgr = cap.read()
            if not success:
                videoWriter.write(img_bgr)
                break
            boxes = infer(yolov8_model, img_bgr)
            draw_bboxes(boxes, img_bgr)
    
            videoWriter.write(img_bgr)

    elif path.endswith(".jpg"):
        img_bgr = cv2.imread(path)
        boxes = infer(yolov8_model, img_bgr)
        draw_bboxes(boxes, img_bgr)
        #write the image as per the original name
        img_name = os.path.splitext(os.path.basename(path))[0]
        output_name = os.path.join("../result", img_name + "-annotated.jpg")
        cv2.imwrite(output_name, img_bgr)

#get the feature map after model inference  
def infer(yolov8_model, img):
    #resize the image
    resized_img = preprocess.resize_image(img, YOLOV8_INPUT_SIZE)
    #get the feature map
    feature_map = yolov8_model.execute([resized_img,]) 
    feature_map = np.array(feature_map)
    # print("=" * 95)
    # print(f"Fetaure Map Values={feature_map}, type={feature_map}")
    # print(f"Shape={feature_map.shape}")
    # print(f"Number of dimensions={feature_map.ndim}")
    # print("=" * 95)
    
    #post-process it with nms
    feature_map = feature_map[0]
    # print(f"shape of feature map=,{feature_map.shape}, total num of elements={feature_map.size}")
    predictions = postprocess.non_max_suppression(feature_map, CONF_THRESHOLD, IOU_THRESHOLD)
    predictions = np.array(predictions)
    # print("=" * 95)
    # print(f"Processed Prediction Values={predictions}, type={predictions}")
    # print(f"Shape={predictions.shape}")
    # print(f"Number of dimensions={predictions.ndim}")
    # print("=" * 95)
    
    # Process detections
    bboxes = []
    for i, det in enumerate(predictions):  # detections per image
        # Rescale boxes from img_size to im0 size
        if det is not None:
            det[:, :4] = postprocess.scale_coords(YOLOV8_INPUT_SIZE, det[:, :4], img.shape).round()
            for *xyxy, conf, cls in det:
                bboxes.append([*xyxy, conf, int(cls)])
        else:
            pass
    # print(f"type of bboxes={type(bboxes)}, bboxes={bboxes}")
    return bboxes

#draw it on the frames
def draw_bboxes(boxes, img):
    
    for bbox in boxes:
        label = utils.coco_map(int(bbox[5]))
        print(f"Bounding box coordinates={bbox[:4]}, confidence={bbox[4]}, class={bbox[5]} label={label}")
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]) + len(label) * 15, int(bbox[1]) + 20), (255, 255, 255), -1)
        cv2.putText(img, label, (int(bbox[0]) + 5, int(bbox[1]) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
def main():
    IMG_PATH="../data/000000000139.jpg"                                 #OK
    MP4_PATH="../data/dude.mp4"                                   #OK
    
    #perform yolov8s detection
    path = MP4_PATH
    object_detect(path, show_debug=False)

if __name__ == '__main__':
    main()
    
