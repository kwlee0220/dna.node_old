FROM python:3.10-slim-bullseye
LABEL name='dna-tools'
LABEL email='kwlee@etri.re.kr'

RUN apt-get update && \
	apt-get -y install libgl1-mesa-glx\
						libsm6 \
						libxext6 \
						libgtk2.0-dev \
						libglib2.0-0 \
						ffmpeg
RUN (apt-get autoremove -y; \
     apt-get autoclean -y)
     
COPY dna.tools-2.1.0-py3-none-any.whl dna.tools-2.1.0-py3-none-any.whl
RUN pip install --upgrade pip && \
    pip install dna.tools-2.1.0-py3-none-any.whl && \
    rm dna.tools-2.1.0-py3-none-any.whl

VOLUME /dna.tools
WORKDIR	/dna.tools

CMD ["/bin/bash"]