# Summer School Computer Vision Exercises


## Installation
```bash
./environment.sh
conda activate summerCV
```

## ROS Setup - Image stream
```
export ROS_IP=192.168.0.100 # hostname -I
export ROS_MASTER_URI=http://192.168.0.102:11311
```

## Dataset Download
```
git lfs install
```
if this returns errors run the following (on linux):
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
```
or (on mac):
```
brew update
brew install git-lfs
```

Check correct install with

```
git lfs
```

Run
```
git lfs pull
```
to pull all images and not just the code.


## plot trajectory
![Image text](https://gitlab.gwdg.de/cns-group/summer_school_cv_exercises/-/raw/main/results/writing8.gif)
![Image text](https://gitlab.gwdg.de/cns-group/summer_school_cv_exercises/-/raw/main/results/writingseb.gif)
