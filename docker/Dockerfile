FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
MAINTAINER kwlee <kwlee@etri.re.kr>

WORKDIR	/dna.node

COPY dna.node-0.0.3-py3-none-any.whl dna.node-0.0.3-py3-none-any.whl
COPY share share
RUN pip install --upgrade pip && \
		pip install dna.node-0.0.3-py3-none-any.whl && \
		rm dna.node-0.0.3-py3-none-any.whl

RUN apt-get update && \
		apt-get install -y libgtk2.0-dev \
											pkg-config \
											libgl1-mesa-glx \
											libglib2.0-0

RUN (apt-get autoremove -y; \
     apt-get autoclean -y)

ENV QT_X11_NO_MITSHM=1

CMD ["dna_node", "share/dna.node.yaml" ]
