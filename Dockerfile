FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Need gcc for pycocotools
RUN apt-get update && apt-get install -y gcc zip htop screen libgl1-mesa-glx libglib2.0-0

# Install pip dependencies
RUN pip install 'azureml-mlflow==1.39.0.post1' \
                'mlflow-skinny==1.26.1' \
                'pytorch-accelerated==0.1.39' \
                'func_to_script' \
                'albumentations==1.2.0' \
                'pandas' \
                'matplotlib>=3.2.2' \
                'pycocotools' \
                'PyYAML>=5.3.1' \
                'scipy>=1.4.1' \
                'ensemble-boxes' \
                'pytest'