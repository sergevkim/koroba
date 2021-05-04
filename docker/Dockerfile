FROM pytorch/pytorch

ENV DEBIAN_FRONTEND=noninteractive
ENV PYOPENGL_PLATFORM osmesa
ENV HOME /root

# open3d dependencies
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    apt-utils \
    build-essential \
    cmake \
    freeglut3 \
    freeglut3-dev \
    g++ \
    git \
    gdb \
    libeigen3-dev \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3-dev \
    libglu1-mesa-dev \
    libosmesa6-dev \
    libpng-dev \
    libqtgui4 \
    libxi-dev \
    llvm-6.0 \
    llvm-6.0-tools \
    lxde \
    mesa-common-dev \
    mesa-utils \
    ne \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-tk \
    python3-dbg \
    pybind11-dev \
    python3-setuptools \
    software-properties-common \
    websockify \
    wget \
    x11-xserver-utils \
    x11vnc \
    xinit \
    xorg-dev \
    xserver-xorg-video-dummy \
    xserver-xorg-input-void \
    xterm \
    xvfb && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 --no-cache-dir install --upgrade pip
RUN pip3 --no-cache-dir install \
    easydict \
    jupyter \
    matplotlib \
    numpy \
    open3d \
    opencv-python-headless \
    pandas \
    plotly \
    physt \
    py7zr \
    pyrender \
    pyquaternion \
    pyunpack \
    scikit-build \
    scikit-image \
    scikit-learn \
    scipy \
    tqdm \
    wheel

ENV HOME /root

WORKDIR $HOME

CMD ["bash"]

