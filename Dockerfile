# FROM continuumio/anaconda3  # seems to be very hard to properly use the GPU

# base doesn't contain the libraries

FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

# parts from anaconda dockerfiles

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

ENV PYTHONDONTWRITEBYTECODE=1

LABEL maintainer=c.sachs@fz-juelich.de

COPY . /tmp/junnbuild/

RUN apt-get update && \
    apt-get install -y wget libglib2.0-0 libnvinfer6 libnvinfer-plugin6 build-essential && \
    CONDA_INSTALLER=https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh && \
    # https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
    wget --quiet $CONDA_INSTALLER -O ~/conda-installer.sh && \
    bash ~/conda-installer.sh -b -p /opt/conda && \
    rm ~/conda-installer.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy && \
    rm -rf /var/lib/apt/lists/* && \
    echo "Finished general CUDA and conda installation ..." && \
    date && \
    cd /tmp/junnbuild && \
    conda config --add channels conda-forge --add channels bioconda --add channels csachs && \
    conda install -y conda-build conda-verify ipython mpich && \
    conda build recipe/junn-predict-recipe && \
    conda install -y -c local junn-predict && \
    conda build recipe && \
    conda install -y -c local junn && \
    # no need for conda cuda installation since it is included in the base image
    # conda install cudatoolkit=10.1.243 cupti cudnn=7.6.5 && \
    conda clean -afy || true && \
    conda build purge-all && \
    pip uninstall -y tunable && pip install https://github.com/csachs/tunable/archive/master.zip && \
    # tunable upgrade can be skipped once new tunable is in anaconda cloud
    pip install tensorflow tensorflow-addons tensorflow-serving-api && \
    pip install opencv-python-headless && \
    pip install horovod && \
    date && \
    echo "Done with JUNN building/installing."

ENTRYPOINT ["python", "-m", "junn"]
