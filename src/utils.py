import acl

#returns the acl data type of an input or output 
def check_data_type(model_desc, i, inp=True):
    
    input_type = acl.mdl.get_input_data_type(model_desc, i) if inp else acl.mdl.get_output_data_type(model_desc, i)
    data_type_mapping = {
    -1: "ACL_DT_UNDEFINED",
    0: "ACL_FLOAT",
    1: "ACL_FLOAT16",
    2: "ACL_INT8",
    3: "ACL_INT32",
    4: "ACL_UINT8",
    6: "ACL_INT16",
    7: "ACL_UINT16",
    8: "ACL_UINT32",
    9: "ACL_INT64",
    10: "ACL_UINT64",
    11: "ACL_DOUBLE",
    12: "ACL_BOOL"
}
    return data_type_mapping[input_type]

#obtains all of the model's inputs and output's
def get_sizes(model_desc, show_model=False):
    input_size = acl.mdl.get_num_inputs(model_desc)
    output_size = acl.mdl.get_num_outputs(model_desc)
    
    if show_model == True:
        print("Total number of inputs in this model:", input_size)
        for i in range(input_size):
            print(f"Model input {i+1} of {input_size}:")
            print("Model input node: ", acl.mdl.get_input_dims(model_desc, i))
            print("Datatype of this input: ", check_data_type(model_desc, i, inp=True))
            model_input_height, model_input_width = (i for i in acl.mdl.get_input_dims(model_desc, i)[0]['dims'][1:3])
        print("=" * 95)
        
        print("Total number of output nodes in this model:", output_size)
        for i in range(output_size):
            print(f"Model output {i+1} of {output_size}:")
            print(f"Model output node {i+1} of {output_size}: ", acl.mdl.get_output_dims(model_desc, i))
            print("Datatype of this output node", check_data_type(model_desc, i, inp=False))
            model_output_height, model_output_width = acl.mdl.get_output_dims(model_desc, i)[0]['dims'][1:]
        print("=" * 95)
        
        return model_input_height, model_input_width, model_output_height, model_output_width
    
    else:
        for i in range(input_size):
            model_input_height, model_input_width = (i for i in acl.mdl.get_input_dims(model_desc, i)[0]['dims'][1:3])
        for i in range(output_size):
            model_output_height, model_output_width = acl.mdl.get_output_dims(model_desc, i)[0]['dims'][1:]
            
    return model_input_height ,model_input_width, model_output_height, model_output_width

#map the value to the label from coco_names
def coco_map(coco_val):
    coco_names = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorbike',
        4: 'aeroplane',
        5: 'bus',
        6: 'train',
        7: 'truck',
        8: 'boat',
        9: 'traffic light',
        10: 'fire hydrant',
        11: 'stop sign',
        12: 'parking meter',
        13: 'bench',
        14: 'bird',
        15: 'cat',
        16: 'dog',
        17: 'horse',
        18: 'sheep',
        19: 'cow',
        20: 'elephant',
        21: 'bear',
        22: 'zebra',
        23: 'giraffe',
        24: 'backpack',
        25: 'umbrella',
        26: 'handbag',
        27: 'tie',
        28: 'suitcase',
        29: 'frisbee',
        30: 'skis',
        31: 'snowboard',
        32: 'sports ball',
        33: 'kite',
        34: 'baseball bat',
        35: 'baseball glove',
        36: 'skateboard',
        37: 'surfboard',
        38: 'tennis racket',
        39: 'bottle',
        40: 'wine glass',
        41: 'cup',
        42: 'fork',
        43: 'knife',
        44: 'spoon',
        45: 'bowl',
        46: 'banana',
        47: 'apple',
        48: 'sandwich',
        49: 'orange',
        50: 'broccoli',
        51: 'carrot',
        52: 'hot dog',
        53: 'pizza',
        54: 'donut',
        55: 'cake',
        56: 'chair',
        57: 'sofa',
        58: 'potted plant',
        59: 'bed',
        60: 'dining table',
        61: 'toilet',
        62: 'tvmonitor',
        63: 'laptop',
        64: 'mouse',
        65: 'remote',
        66: 'keyboard',
        67: 'cell phone',
        68: 'microwave',
        69: 'oven',
        70: 'toaster',
        71: 'sink',
        72: 'refrigerator',
        73: 'book',
        74: 'clock',
        75: 'vase',
        76: 'scissors',
        77: 'teddy bear',
        78: 'hair drier',
        79: 'toothbrush'
    }

    return coco_names[coco_val]