from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")

rf = Roboflow(api_key=API_KEY)
project = rf.workspace("school-95f9t").project("human-detection-uerkn")
version = project.version(3)
dataset = version.download("yolov11")               