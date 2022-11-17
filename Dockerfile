FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Need gcc for pycocotools
RUN apt-get update && apt-get install -y gcc zip htop screen libgl1-mesa-glx libglib2.0-0

COPY requirements.txt /requirements.txt

# Install pip dependencies
RUN pip install -r /requirements.txt