# ORB-SLAM3 stereo-inertial, headless, for Trinet stereo recordings.
# Ubuntu 20.04 + OpenCV 4.2 (distro) + Pangolin v0.6 + ORB_SLAM3 master.
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
        libeigen3-dev libopencv-dev libboost-serialization-dev libssl-dev \
        libglew-dev libgl1-mesa-dev libegl1-mesa-dev libepoxy-dev \
        libpython3-dev python3-distutils \
    && rm -rf /var/lib/apt/lists/*

# Pangolin v0.6 (last release that builds cleanly on 20.04's toolchain)
RUN git clone --depth 1 --branch v0.6 \
        https://github.com/stevenlovegrove/Pangolin.git /tmp/pangolin \
    && cmake -S /tmp/pangolin -B /tmp/pangolin/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_PANGOLIN_PYTHON=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF \
    && cmake --build /tmp/pangolin/build -j8 --target install \
    && ldconfig && rm -rf /tmp/pangolin

RUN git clone --depth 1 https://github.com/UZ-SLAMLab/ORB_SLAM3.git /ORB_SLAM3

WORKDIR /ORB_SLAM3
# 20.04 ships OpenCV 4.2; master's CMakeLists asks for 4.4 — relax it.
RUN sed -i 's/find_package(OpenCV 4.4)/find_package(OpenCV 4.2)/' CMakeLists.txt || true
# build_examples pass: build.sh builds Thirdparty (DBoW2, g2o, Sophus) then core.
RUN sed -i 's/make -j[0-9]*/make -j8/g; s/make -j$/make -j8/g' build.sh \
    && bash build.sh
RUN cd Vocabulary && tar -xzf ORBvoc.txt.tar.gz

ENV LD_LIBRARY_PATH=/ORB_SLAM3/lib:/ORB_SLAM3/Thirdparty/DBoW2/lib:/ORB_SLAM3/Thirdparty/g2o/lib:/usr/local/lib
CMD ["/bin/bash"]
