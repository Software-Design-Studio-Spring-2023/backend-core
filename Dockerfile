# Use miniconda3 as the base image
FROM continuumio/anaconda3:latest

# Set the Python version
ARG PYTHON_VERSION=3.9.16

COPY ./ ./app

# RUN sudo apt install libblas3 liblapack3 liblapack-dev libblas-dev gfortran

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc python-dev libzmq3-dev libgl1-mesa-glx && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install dlib 

# Create a conda environment with the specific Python version
RUN conda env create -f app/environment.yml



# Expose port for your app (modify if using a different port)
EXPOSE 8080

# Activate the environment. Future commands will run in this environment.
SHELL ["conda", "run", "-n", "flask-server", "/bin/bash", "-c"]

RUN echo "Make sure environment is activated:"
RUN conda env list

# Run server
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "flask-server", "python", "app/main.py"]

