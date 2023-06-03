# FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
LABEL name='dna-node'
LABEL email='kwlee@etri.re.kr'

RUN apt-get update && \
		apt-get -y install libgl1-mesa-glx\
							ffmpeg \
							libsm6 \
							libxext6 \
							libgtk2.0-dev \
							libglib2.0-0 \
							ffmpeg
RUN (apt-get autoremove -y; \
     apt-get autoclean -y)

COPY dna.node-2.1.0-py3-none-any.whl dna.node-2.1.0-py3-none-any.whl
RUN pip install --upgrade pip && \
		pip install dna.node-2.1.0-py3-none-any.whl && \
		rm dna.node-2.1.0-py3-none-any.whl

VOLUME /dna.node
VOLUME /root/.cache/torch/hub

WORKDIR	/dna.node

# -e DISPLAY=<local-ip>:0.0
# docker run -it --gpus all -e DISPLAY=192.168.0.15:0.0 -v C:\Temp\dockers\dna\data\dna-node\data:/dna.node/data dna/dna-node /bin/bash