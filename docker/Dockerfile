FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel

RUN apt-get update \
  && apt-get install -y sudo tmux tree vim wget language-pack-ja

#locale
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

# user
ENV USER_NAME=developer
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/${USER_NAME}
RUN chmod u+s /usr/sbin/useradd \
  && chmod u+s /usr/sbin/groupadd
WORKDIR /build
ADD ./entrypoint.sh /build

# conda
ENV PYTHONPATH=/work/src
ADD ./base.yml /build
RUN conda env update -f base.yml \
  && conda init bash

ENTRYPOINT ["/build/entrypoint.sh"]
CMD ["bash"]
