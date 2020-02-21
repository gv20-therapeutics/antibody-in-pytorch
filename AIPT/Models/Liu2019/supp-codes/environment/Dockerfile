FROM registry.codeocean.com/codeocean/tensorflow:1.4.0-python2.7.12-cuda8.0.61-cudnn6.0.21-ubuntu16.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      "git-all=1:2.7.4-0ubuntu1.6" \
      "openssh-client=1:7.2p2-4ubuntu2.6" \
      "screen=4.3.1-2build1" \
      "sshpass=1.05-1" \
      "vim=2:7.4.1689-3ubuntu1.2" \
      "wget=1.17.1-1ubuntu1.4" \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade --no-cache-dir --upgrade-strategy=only-if-needed \
      Cython==0.29.2 \
      Keras==2.2.4 \
      Theano==1.0.3 \
      colorcet==0.9.1 \
      h5py==2.8.0 \
      hyperas==0.4 \
      hyperopt==0.1.1 \
      ipython==5.8.0 \
      jupyter==1.0.0 \
      numpy==1.13.3 \
      seaborn==0.7.1

COPY postInstall /
RUN /postInstall
