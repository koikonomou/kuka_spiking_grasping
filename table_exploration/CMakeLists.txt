cmake_minimum_required(VERSION 3.0.2)
project(table_exploration)


find_package(catkin REQUIRED
	 kuka_gazebo
	 kuka_description
	 kuka_kr4_moveit
	 kuka_kr4_support)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES table_exploration
#  CATKIN_DEPENDS 
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
# ${catkin_INCLUDE_DIRS}
)
