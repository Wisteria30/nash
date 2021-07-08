FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel

RUN apt-get update \
  && apt-get install -y sudo tmux tree vim wget

# user
ENV USER_NAME=developer
RUN echo "${USER_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/${USER_NAME}
RUN chmod u+s /usr/sbin/useradd \
  && chmod u+s /usr/sbin/groupadd
WORKDIR /build
ADD ./entrypoint.sh /build

ENTRYPOINT ["/build/entrypoint.sh"]
CMD ["bash"]
