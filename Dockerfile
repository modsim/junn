FROM nvidia/cuda:11.2.1-cudnn8-runtime-ubuntu20.04

LABEL maintainer=c.sachs@fz-juelich.de

ARG DEBIAN_FRONTEND="noninteractive"
ARG MICROMAMBA="http://api.anaconda.org/download/conda-forge/micromamba/0.8.0/linux-64/micromamba-0.8.0-he9b6cbd_0.tar.bz2"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=UTC
ENV PATH=/opt/conda/bin:${PATH}
ENV PATH=/opt/conda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

ENV PYTHONDONTWRITEBYTECODE=1

ENV CONDA_PREFIX=/opt/conda
ENV MAMBA_ROOT_PREFIX=${CONDA_PREFIX}

COPY . /tmp/junnbuild

WORKDIR /tmp/junnbuild

RUN ln -s libcusolver.so.11 /usr/local/cuda/lib64/libcusolver.so.10 && \
    apt-get update && \
    apt-get install -y --no-install-recommends libglib2.0-0 libgl1-mesa-glx ca-certificates wget build-essential cmake && \
    rm -rf /var/lib/apt/lists/* && \
    WORKDIR=`pwd` && mkdir -p /opt/conda/bin && cd /opt/conda/bin && \
    wget -qO- $MICROMAMBA | tar xj bin/micromamba --strip-components=1 && unset MICROMAMBA && \
    micromamba install -p $MAMBA_ROOT_PREFIX \
        python=3.8 conda \
        keras_nvidia_statistics conda-build conda-verify boa ipython mpich \
        -c conda-forge -c modsim -c csachs && \
    echo "channels:\n- conda-forge\n- modsim\n- csachs" > ~/.condarc && \
    cd $WORKDIR && \
    conda mambabuild junn-predict/recipe && \
    conda mambabuild recipe && \
    mamba install -y -c local junn && \
    mkdir /opt/conda/packages-built && \
    find /opt/conda/conda-bld -name '*.tar.bz2' -exec cp {} /opt/conda/packages-built \; && \
    conda clean -afy || true && \
    conda build purge-all && \
    pip install -v tensorflow tensorflow-addons tensorflow-serving-api && \
    pip install -v horovod && \
    pip cache purge && \
    apt-get purge --autoremove -y wget build-essential cmake && \
    mkdir /data && \
    echo "Done with JUNN building/installing."

WORKDIR /data

ENTRYPOINT ["python", "-m", "junn"]
