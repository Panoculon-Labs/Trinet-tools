FROM osrf/ros:noetic-desktop-full

# Self-contained OpenVINS build (stereo-inertial VIO, ov_msckf + ov_eval).
# The upstream Dockerfile_ros1_20_04 is a dev shell that expects the source
# mounted; this image bakes a pinned clone + catkin build so it can run
# headless on a recording bag with no host setup.
RUN apt-get update && apt-get install -y \
    libeigen3-dev git python3-catkin-tools python3-osrf-pycommon \
    libceres-dev && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /catkin_ws/src && cd /catkin_ws/src && \
    git clone --depth 1 https://github.com/rpng/open_vins.git

RUN cd /catkin_ws && . /opt/ros/noetic/setup.sh && \
    catkin config --init --extend /opt/ros/noetic && \
    catkin build -j6 --no-status ov_msckf ov_eval

RUN echo 'source /catkin_ws/devel/setup.bash' >> /root/.bashrc
