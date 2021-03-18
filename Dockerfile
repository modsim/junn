FROM nvidia/cuda:11.2.1-cudnn8-runtime-ubuntu20.04

LABEL maintainer=c.sachs@fz-juelich.de

ARG DEBIAN_FRONTEND="noninteractive"
ARG MICROMAMBA="https://anaconda.org/conda-forge/micromamba/0.8.2/download/linux-64/micromamba-0.8.2-1.tar.bz2"
ARG PYTHONVERSION=3.8

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=UTC
ENV PATH=/opt/conda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

ENV PYTHONDONTWRITEBYTECODE=1

ENV CONDA_PREFIX=/opt/conda
ENV MAMBA_ROOT_PREFIX=${CONDA_PREFIX}

COPY . /tmp/package

WORKDIR /tmp/package

RUN BUILD_START=`date +%s%N` && \
    apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates wget build-essential cmake libglib2.0-0 libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/* && \
    WORKDIR=`pwd` && mkdir -p /opt/conda/bin && cd /opt/conda/bin && \
    wget -qO- $MICROMAMBA | tar xj bin/micromamba --strip-components=1 && unset MICROMAMBA && \
    micromamba install -p $MAMBA_ROOT_PREFIX \
        python=$PYTHONVERSION conda conda-build conda-verify boa \
        -c conda-forge && \
    echo "channels:\n- conda-forge\n- modsim\n" > ~/.condarc && \
    cd $WORKDIR && \
    conda mambabuild junn-predict/recipe && conda mambabuild recipe && \
    mamba install -y -c local junn ipython mpich && \
    mkdir /opt/conda/packages && \
    find /opt/conda/conda-bld -name '*.tar.bz2' -exec cp {} /opt/conda/packages \; && \
    conda clean -afy || true && \
    conda build purge-all && \
    pip install -v tensorflow tensorflow-addons tensorflow-serving-api && pip install -v horovod && \
    pip cache purge || true && \
    apt-get purge --autoremove -y wget build-essential cmake && \
    useradd -m user && \
    mkdir /data && \
    chown -R user:user /data && \
    ln -s libcusolver.so.11 /usr/local/cuda/lib64/libcusolver.so.10 && \
    BUILD_FINISH=`date +%s%N` && \
    echo "Build done, took `perl -e "print(($BUILD_FINISH-$BUILD_START)/1000000000)"` seconds."

USER user

WORKDIR /data

ENTRYPOINT ["python", "-m", "junn"]
