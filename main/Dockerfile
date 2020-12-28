FROM continuumio/miniconda3

WORKDIR /app
# Create the environment:
COPY . .
RUN conda env create -n myenv --file=requirements.yml 
RUN apt-get update 
RUN apt-get install ffmpeg libsm6 libxext6  -y
ENTRYPOINT /opt/conda/bin/conda run -n myenv python main.py /data/inputs /data/outputs
