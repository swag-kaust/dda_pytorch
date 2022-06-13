FROM nvcr.io/nvidia/pytorch:22.03-py3

ARG MYUSER=this_user
ARG MYUID=1000
ARG MYGID=1000

ENV DEBIAN_FRONTEND noninteractive

# Install cmake, add user
RUN mkdir -p /workspace && cd /workspace && \
    if [ -f /usr/bin/yum ]; then yum install -y make wget vim; fi && \
    if [ -f /usr/bin/apt-get ]; then apt-get update && apt-get install -y apt-utils make wget vim; fi && \
    groupadd -f -g ${MYGID} ${MYUSER} && \
    useradd -rm -u $MYUID -g $MYUSER -p "" $MYUSER && \
    chown ${MYUSER}:${MYGID} /workspace 

USER $MYUSER

RUN pip install ipywidgets==7.7.0

WORKDIR /workspace/project

