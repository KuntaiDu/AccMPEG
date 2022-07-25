import logging
import os

import coloredlogs
import numpy as np
from openvino.inference_engine import Blob, IECore, TensorDesc

from timer import Timer

coloredlogs.install(
    fmt="%(asctime)s [%(levelname)s] %(name)s:%(funcName)s[%(lineno)s] -- %(message)s",
    level="INFO",
)


logger = logging.getLogger("AccMPEG")

os.system("rm AccMPEG.log")
fh = logging.FileHandler("AccMPEG.log")
logger.addHandler(fh)

# load model
XML_PATH = "./onnx/accmpeg.xml"
BIN_PATH = "./onnx/accmpeg.bin"
ie_core_handler = IECore()
network = ie_core_handler.read_network(model=XML_PATH, weights=BIN_PATH)
executable_network = ie_core_handler.load_network(
    network, device_name="CPU", num_requests=100
)

# define input data, resolution: 720p
random_input_data = np.random.randn(1, 3, 720, 1280).astype(np.float32)
tensor_description = TensorDesc(
    precision="FP32", dims=(1, 3, 720, 1280), layout="NCHW"
)
input_blob = Blob(tensor_description, random_input_data)

# perform inference for 100 times
for i in range(100):
    with Timer("AccMPEG", logger):  # measure the time
        inference_request = executable_network.requests[i]
        inference_request.set_blob(blob_name="input", blob=input_blob)
        inference_request.infer()   # inference
        output = inference_request.output_blobs["output"].buffer


import re

import numpy as np

pattern = r"([0-1].*)"
pattern = re.compile(pattern)
times = []

with open("AccMPEG.log", "r") as f:
    for val in re.findall(pattern, f.read()):
        times.append(float(val))

print("AccMPEG runs for ", np.mean(times), " sec on a 720p frame in average.")

