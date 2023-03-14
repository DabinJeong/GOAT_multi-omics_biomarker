# Table of contents
* [Project description](#Project-description)
* [Setup](#setup)
* [Run](#run)

# Project description
![workflow](./img/method_overview.pdf)  
We propose a novel deep attention model for biomarker discovery for the asthma subtype by incorporating complex interactions between biomolecules in the deep learning model and capturing key biomarker candidates using the attention mechanism.


# Setup
## Build docker image
~~~
docker build --tag biomarker:0.1.0 .
~~~
## Install workflow manager: Nextflow
~~~
conda create -n biomarker python=3.9
conda activate biomarker
conda install -c bioconda nextflow=21.04.0
~~~

# Run
~~~
nextflow run biomarker_discovery.nf -c pipeline.config -with-docker biomarker:0.1.0
~~~
