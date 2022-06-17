# Spiking RL package

In order to run this package you need to have Ubuntu 18.04 alongwith ROS Melodic!
You need :
```
	python 2 
	python 3.7
```
In order to switch between python versions use:
```
	sudo update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1 
	sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.7 2 
```
Type the above command to choose python3 version before installing conda:
```
	sudo update-alternatives --config python 
```

Next install cuda and conda package:
```
	[Cuda package](https://developer.nvidia.com/cuda-downloads) 
	[Anacoda package](https://www.anaconda.com/products/distribution#Downloads)
```
Create a conda env in order to run python3 experiments with ROS Melodic and run :
```
	conda create --name snn_train python=3.7 
	conda activate snntrain 
	conda install pytorch tensorflow tensorboard 
	sudo apt install -y python3 python3-dev python3-pip build-essential 
	sudo -H pip3 install rosdep rospkg rosinstall_generator rosinstall wstool vcstools catkin_tools catkin_pkg 
	pip3 install rospy 
	pip install setuptools==59.5.0 
	pip install shapely 
```