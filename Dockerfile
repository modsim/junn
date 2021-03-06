# FROM continuumio/anaconda3  # seems to be very hard to properly use the GPU

# base doesn't contain the libraries

FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

# parts from anaconda dockerfiles

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

ENV PYTHONDONTWRITEBYTECODE=1

LABEL maintainer=c.sachs@fz-juelich.de

COPY . /tmp/junnbuild/

WORKDIR /tmp/junnbuild

RUN apt-get update && \
    apt-get install -y wget libglib2.0-0 libnvinfer6 libnvinfer-plugin6 build-essential libgl1-mesa-glx && \
    CONDA_INSTALLER=https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh && \
    wget --quiet $CONDA_INSTALLER -O ~/conda-installer.sh && \
    bash ~/conda-installer.sh -b -p /opt/conda && \
    rm ~/conda-installer.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    conda config --add channels conda-forge && \
    conda update --all && \
    conda install -y conda-build conda-verify && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    conda clean -afy && \
    rm -rf /var/lib/apt/lists/* && \
    echo "Starting stage two" && \
    apt-get update && \
    apt-get install -y cmake && \
    rm -rf /var/lib/apt/lists/* && \
    conda config --add channels modsim && \
    conda config --add channels csachs && \
    conda install -y ipython mpich python=3.8 && \
    conda build junn-predict/recipe recipe && \
    conda install -y -c local junn && \
    # no need for conda cuda installation since it is included in the base image
    # conda install cudatoolkit=10.1.243 cupti cudnn=7.6.5 && \
    mkdir /opt/junn && \
    find /opt/conda/conda-bld -name '*.tar.bz2' -exec cp {} /opt/junn && \
    conda clean -afy || true && \
    conda build purge-all && \
    pip install tensorflow tensorflow-addons tensorflow-serving-api && \
    pip install horovod && \
    mkdir /data && \
    date && \
    echo "Done with JUNN building/installing."

WORKDIR /data

ENTRYPOINT ["python", "-m", "junn"
