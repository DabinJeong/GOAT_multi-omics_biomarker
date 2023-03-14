FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04 

LABEL maintainer="Dabin Jeong"

ENV TZ=Asia/Seoul
ENV DEBCONF_NOWARNINGS yes
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y apt-utils && apt-get install -y wget
RUN apt-get install -y libzmq3-dev libharfbuzz-dev libfribidi-dev libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev build-essential libcurl4-openssl-dev libxml2-dev libssl-dev libfontconfig1-dev 
## install R
RUN echo "deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/" >> /etc/apt/sources.list
RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc
RUN apt-get update
RUN apt-get install -y r-base
RUN Rscript -e "if (!require('BiocManager', quietly = TRUE)) install.packages('BiocManager')"
RUN Rscript -e "install.packages('argparse', dependencies = TRUE)"
RUN Rscript -e "install.packages('tidyverse',dependencies = TRUE)"
RUN PATH=/usr/bin/R:$PATH;export PATH

### Install miniconda
#ENV PATH /opt/conda/bin:$PATH
#RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-py37_4.11.0-Linux-x86_64.sh -O /tmp/miniconda.sh && \
#    echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
#    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
#    rm -rf /tmp/*
#
### Install mamba
#RUN conda update -n base -c defaults conda
#RUN conda install -y mamba -c conda-forge
#
#ADD ./environment.yml .
#RUN mamba env update --file ./environment.yml &&\
#    conda clean -tipy
#
#RUN conda init bash 
#
### Install torch, torch-geometric
#RUN pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip install torch-scatter torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.8.1+cu101.html
#RUN pip install torch-geometric

COPY modules/* /tools/

ENTRYPOINT ["/usr/bin/env"]
CMD ["/bin/bash"]
