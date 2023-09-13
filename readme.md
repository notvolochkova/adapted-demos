# Person or Object Detection
This code collection's purpose is to draw bounding boxes
on any persons or other objects captured in the video or image formats. The code has been tested `.mp4` and `.jpg`
images

## YOLOv8s
Pre-trained model used is `yolov8s.pt` obtained from
Ultralytics. Check out their page on [github](https://github.com/ultralytics/ultralytics).

This model is not fine-tuned.

## Execution
Executed on the host machine, the directory structure should be:
```
(base) $: tree -L 2 .
.
├── acllite
│   ├── acllite_imageproc.py
│   ├── acllite_image.py
│   ├── acllite_logger.py
│   ├── acllite_model.py
│   ├── acllite_resource.py
│   ├── acllite_utils.py
│   ├── cameracapture.py
│   ├── constants.py
│   ├── dvpp_vdec.py
│   ├── __init__.py
│   ├── lib
│   ├── presenteragent
│   ├── __pycache__
│   └── videocapture.py
├── config
│   ├── aipp_cfg
│   └── coco.names
├── data
│   ├── 000000000139.jpg
│   ├── 000000000285.jpg
│   ├── 000000000632.jpg
│   ├── 000000000724.jpg
│   ├── 000000000776.jpg
│   ├── bottles.jpg
│   ├── dog.jpg
│   └── orange.jpg
├── models
│   ├── convert.py
│   ├── yolov8s.om
│   ├── yolov8s.onnx
│   └── yolov8s.pt
├── requirements
├── result
│   ├── 000000000139-annotated.jpg
│   ├── 000000000285-annotated.jpg
│   ├── 000000000632-annotated.jpg
│   ├── 000000000724-annotated.jpg
│   ├── 000000000776-annotated.jpg
│   ├── bottles-annotated.jpg
│   ├── dog-annotated.jpg
│   └── orange-annotated.jpg
└── src
    ├── acl_yolov8.py
    ├── postprocess.py
    ├── preprocess.py
    ├── __pycache__
    └── utils.py
```

### How to setup the environment?
1. Clone this repo
2. Enter the directory
3. The entrypoint of program is in `src` a file called `acl_yolov8.py`
4. Set your environment variable with `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
```
(base) $: source /usr/local/Ascend/ascend-toolkit/set_env.sh
```
5. Check if you can invoke `atc` and `python -c "import acl"`. If yes, you can proceed. If not check your environment.
```
(base) $: atc
ATC start working now, please wait for a moment.
ATC run failed, Please check the detail log, Try 'atc --help' for more information
E10054: The requied parameter [--model] for ATC is empty. Another possible reason is that the value of some parameter is not enclosed by quotation marks ("").
```
```
(base) $: python -c "import acl"
```
5. Activate your conda environment `conda activate <name>`
```
(base) $: conda activate acl_yolov8
```
6. Install requirements `pip install -r requirements`
```
(acl_yolov8) $: pip install -r requirements
Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: onnx in /usr/local/miniconda3/envs/acl_yolov8/lib/python3.8/site-packages (from -r requirements (line 1)) (1.14.1)
Requirement already satisfied: ultralytics in /usr/local/miniconda3/envs/acl_yolov8/lib/python3.8/site-packages (from -r requirements (line 2)) (8.0.175)
```
7. Enter directory `models` edit the python code to convert your `.pt` eventually to `.om`.
8. Run the code as `python convert.py`
9. Enter directory `src` and start the program with  `python acl_yolov8.py`

## TODO
1. Test on A500-3000 ✅
2. msquickcmp ✅
3. Performance improvement, check mxBase repo 🟡


## Work contribution
This contribution of work is done by the regional Ascend FAE from Huawei UAE