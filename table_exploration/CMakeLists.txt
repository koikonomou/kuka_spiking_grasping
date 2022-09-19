cmake_minimum_required(VERSION 3.0.2)
project(table_exploration)
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
	 # kuka_gazebo
	 message_filters
	 visualization_msgs
	 kuka_kr4_moveit
	 kuka_kr4_support
	 message_generation
	 roscpp
	 sensor_msgs
	 std_msgs
	 )

find_package(cmake_modules REQUIRED)
find_package(Boost COMPONENTS format)

add_message_files(FILES 
	Distance.msg
	Collision.msg)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES table_exploration
	CATKIN_DEPENDS roscpp message_runtime
#  DEPENDS system_lib
)
###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include 
	${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

add_executable(msg_syn src/msg_syn.cpp)
target_link_libraries(msg_syn ${catkin_LIBRARIES})