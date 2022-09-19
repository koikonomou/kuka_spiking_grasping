#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/JointState.h>
#include <visualization_msgs/MarkerArray.h>
#include <table_exploration/Distance.h>
#include <table_exploration/Collision.h>

using namespace sensor_msgs;
using namespace message_filters;
using namespace table_exploration;
using namespace visualization_msgs;

ros::Publisher scan_pub;
ros::Publisher col_pub;
ros::Publisher camera_pub;
ros::Publisher marker_pub;
ros::Publisher joint_states_pub;


void callback(const Distance::ConstPtr& scan_pub1, const Collision::ConstPtr& col_pub1, const Distance::ConstPtr& camera_pub1, const JointState::ConstPtr& joint_states_pub1)
{
  // Solve all of perception here...
  ROS_INFO("Sync_Callback");

  scan_pub.publish(scan_pub1);
  col_pub.publish(col_pub1);
  camera_pub.publish(camera_pub1);
  joint_states_pub.publish(joint_states_pub1);

}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "msg_syn");

  std::string scan_topic = "/kuka/collision";
  std::string col_topic = "/collision_detection";
  std::string camera_topic = "/kuka/box/distance";
  std::string joint_states_topic = "/joint_states";


  ros::NodeHandle nh;
  message_filters::Subscriber<Distance> scan_sub(nh, scan_topic, 100);
  message_filters::Subscriber<Collision> col_sub(nh, col_topic, 100);
  message_filters::Subscriber<Distance> camera_sub(nh, camera_topic, 100);
  message_filters::Subscriber<JointState> joint_state_sub(nh, joint_states_topic, 100);

  scan_pub = nh.advertise<table_exploration::Distance>("/synchronized" + scan_topic, 1000);
  col_pub = nh.advertise<table_exploration::Collision>("/synchronized" + col_topic, 1000);
  camera_pub = nh.advertise<table_exploration::Distance>("/synchronized" + camera_topic, 1000);
  joint_states_pub = nh.advertise<sensor_msgs::JointState>("/synchronized" + joint_states_topic, 1000);



  typedef sync_policies::ApproximateTime<Distance, Collision, Distance, JointState> MySyncPolicy;
  // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), scan_sub, col_sub, camera_sub, joint_state_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3, _4));


  ros::spin();

  return 0;
}