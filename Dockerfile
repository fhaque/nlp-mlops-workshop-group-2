FROM continuumio/miniconda:latest

# Set Docker's work directory to root
WORKDIR /root

# Copy over model and API code into the container
COPY ./server /root
COPY ./model /root/model

# Create a conda environment from the specified conda.yaml
RUN conda env create --file /root/model/conda.yaml

# Add to .bashrc
RUN echo "source activate mlflow-env" >> .bashrc

# Pip install api requirements into the conda env
RUN /bin/bash -c "source activate mlflow-env && python -m pip install --upgrade pip setuptools && python -m pip install -r /root/requirements.txt --no-cache-dir"

# Make our start script executable
RUN ["chmod", "+x", "/root/start.sh"]

# Start the API
ENTRYPOINT [ "/root/start.sh" ]
