eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda create --name summerCV --yes python=3.9
conda activate summerCV

# You need pytorch version 2.0
# For example:
#pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

if [ "$(uname)" == "Darwin" ]; then
    # for mac users
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
else
    # for windows/linux users
    pip install torch==2.0.1+cpu torchvision==0.15.2+cpu torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
fi

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
mim install "mmpose>=1.1.0"

pip install jsonpickle
pip install opencv-python
pip install natsort

pip install --extra-index-url https://rospypi.github.io/simple/ rospy
pip install --extra-index-url https://rospypi.github.io/simple/ cv_bridge
pip install --extra-index-url https://rospypi.github.io/simple/ message_filters
pip install --extra-index-url https://rospypi.github.io/simple/ sensor_msgs
pip install --extra-index-url https://rospypi.github.io/simple/ geometry_msgs
