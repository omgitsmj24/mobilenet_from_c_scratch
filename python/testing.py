import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from IPython.display import display
import sys
import pandas as pd

# np.set_printoptions(threshold=sys.maxsize)   

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="/home/hoangth34/mobilenet_from_c_scratch/model/v3-large-minimalistic_224_1.0_uint8.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
image_path = "sample_data/Snail.jpg"
input_shape = input_details[0]['shape']
image = Image.open(image_path)
image = image.resize((224,224))
input_data_pil = np.expand_dims(np.array(image), 0)
interpreter.set_tensor(input_details[0]['index'], input_data_pil)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])

# Get Labels
with open("mobilenet_labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]
len(labels)

# Print result
print("max value of output_data: ", output_data.max())
print("Index of output_data: ", max(output_data).argmax())
print("Label of output_data: ", labels[max(output_data).argmax()])