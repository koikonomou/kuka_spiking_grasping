import os

def generate_negative_description_file():
  with open ('neg.txt','w') as f:
    for filename in os.listdir('/home/katerina/catkin_ws/src/kuka_spiking_grasping/table_exploration/negative_images/negatives'):
      f.write('negatives/'+filename+'\n')

