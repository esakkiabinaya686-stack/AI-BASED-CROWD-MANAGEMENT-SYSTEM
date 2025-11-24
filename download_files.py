import urllib.request
import os

def download_yolo_files():
    files = {
        'yolov3.weights': 'https://pjreddie.com/media/files/yolov3.weights',
        'yolov3.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
        'coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'
    }
    
    for filename, url in files.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        else:
            print(f"{filename} already exists")

# Download the YOLO files first
download_yolo_files()