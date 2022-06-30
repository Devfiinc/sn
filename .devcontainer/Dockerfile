# ==================================================================
# module list
# ------------------------------------------------------------------
# Ubuntu           20.04
# OpenMPI          latest       (apt)
# cmake            3.16.3       (apt)
# MKL              2020.4-912   (apt)
# arrayfire        3.7.3        (git, CPU backend)
# libsndfile       latest       (apt)
# oneDNN           v2.0         (git)
# Gloo             1da2117      (git)
# FFTW             latest       (apt)
# KenLM            0c4dd4e      (git)
# GLOG             latest       (apt)
# gflags           latest       (apt)
# python3          latest       (apt)
# ==================================================================

#############################################################################
#                             APT IMAGE + CMAKE                             #
#############################################################################

#FROM ubuntu:20.04 as cpu_base_builder
FROM ubuntu:20.04 as cpu_base_builder
#FROM postgres:latest

ENV APT_INSTALL="apt-get install -y --no-install-recommends"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        wget \
        ssh \
        git \
        g++ \
        cmake \
        sudo \
        software-properties-common \
        python3-dev \
        python3-pip \
        libpq-dev \
        curl \
        mlocate \
        nano


RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"


# Install Python 3.9
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.9 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2 && \
    update-alternatives --config python3 && \
    apt -y install python3.9-dev python3.9-venv

# Clean up
RUN apt-get clean && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/*

# Install tensorflow-federated and postgres
RUN pip install --upgrade pip && \
    pip install --upgrade tensorflow-federated && \
    pip install --upgrade psycopg2

RUN pip3 install -U numpy \ 
                scipy \ 
                joblib \ 
                threadpoolctl \ 
                cython \ 
                matplotlib \ 
                scikit-image \ 
                pandas \ 
                seaborn \ 
                memory_profiler \ 
                pytest \ 
                pytest-cov \ 
                flake8 \ 
                black \ 
                pyamg \ 
                sphinx \ 
                sphinx-gallery \ 
                numpydoc \ 
                Pillow \ 
                sphinx-prompt \ 
                sphinxext-opengraph \
                torch \
                scikit-learn \
                pickle-mixin
    
#RUN apt -y update && apt -y install make libreadline-dev bison flex pkg-config libssl-dev libclang-dev clang
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        make \
        libreadline-dev \
        bison \
        flex \
        pkg-config \
        libssl-dev \
        libclang-dev \
        clang

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y -qq --fix-missing
RUN apt-get install -y wget gnupg
RUN echo "deb http://apt.postgresql.org/pub/repos/apt/ focal-pgdg main" >> /etc/apt/sources.list.d/pgdg.list
RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -
RUN apt update

RUN apt-get update -y --fix-missing && \
    apt-get install -y clang-10 \
                       llvm-10 \
                       gcc \
                       libz-dev \
                       zlib1g-dev \
                       strace \
                       libssl-dev
# RUN apt-get install -y postgresql-10 postgresql-server-dev-10
# RUN apt-get install -y postgresql-11 postgresql-server-dev-11
# RUN apt-get install -y postgresql-12 postgresql-server-dev-12
# RUN apt-get install -y postgresql-13 postgresql-server-dev-13

RUN apt-get install -y postgresql-14 \
                       postgresql-server-dev-14

RUN apt-get install -y ruby \
                       ruby-dev \
                       rubygems
                       
RUN gem install --no-document fpm






ARG USER=docker
ARG UID=1000
ARG GID=1000

RUN useradd -m ${USER} --uid=${UID}

USER ${UID}:${GID}
WORKDIR /home/${USER}
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y
ENV PATH="/home/${USER}/.cargo/bin:${PATH}"

RUN cargo install cargo-pgx
RUN cargo pgx init

RUN cargo pgx init \
#	--pg10=/usr/lib/postgresql/10/bin/pg_config \
#	--pg11=/usr/lib/postgresql/11/bin/pg_config \
#	--pg12=/usr/lib/postgresql/12/bin/pg_config \
#	--pg13=/usr/lib/postgresql/13/bin/pg_config \
	--pg14=/usr/lib/postgresql/14/bin/pg_config


USER root
WORKDIR /



RUN pip3 install -U opendp \
                    smartnoise-sql \
                    diffprivlib \
                    seaborn