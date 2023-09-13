import os
import subprocess                       
#from ultralytics import YOLO  

#TODO: edit this to handle cwd
def extract_filename(file_path):
    
    directory, file_name = os.path.split(file_path)
    base_name, extension = os.path.splitext(file_name)

    return directory, base_name

converts the pt model into onnx
def pt2onnx():
   PT_PATH="../models/yolov8s.pt"
   try:
       model = YOLO(PT_PATH)
       model.export(format='onnx', opset=11, dynamic=False, simplify=False)
       print(f"Success in exporting from .pt -> .onnx {extract_filename(PT_PATH)[0]}")

   except Exception as e:
       print(f"Failure in exporting from .pt -> .onnx: {e}")

#converts onnx into om        
def onnx2om():
    ONNX_PATH="../models/yolo8s.onnx"
    ATC_COMMAND="""
    atc --model=../models/yolov8s.onnx \
    --framework=5 \
    --output=../models/yolov8s_noaipp \
    --input_format=NCHW \
    --soc_version=Ascend310 \
    --input_shape="images:1,3,640,640" \
    --enable_small_channel=1 \
"""
    
    try:
        subprocess.run(ATC_COMMAND, shell=True, check=True, executable='bash')
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
    except Exception as e:
        print(f"Another problem; {e}")

def main():
   pt2onnx() 
   onnx2om() 

if __name__ == '__main__':
    main()
