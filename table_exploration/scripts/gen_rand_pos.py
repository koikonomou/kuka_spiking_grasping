import math
import rospy
import numpy as np
from std_msgs.msg import Float64

#joint a1 = [-2.97 , 2.97]
#joint a2 = [-3.40 , 0.7]
#joint a3 = [-2.01 , 2.62]
#joint a4 = [-3.23 , 3.23]
#joint a5 = [-2.09 , 2.09]
#joint a6 = [-6.11, 6.11]

j_a1_init = -1.47988049
j_a2_init = 0.20534724
j_a3_init = 0.06428167
j_a4_init = 0.03506831
j_a5_init = -0.06441653
j_a6_init = 0

j_a1 = random.uniform(-2.97 , 2.97)
j_a2 = random.uniform(-3.40 , 0.7)
j_a3 = random.uniform(-2.01 , 2.62)
j_a4 = random.uniform(-3.23 , 3.23)
j_a5 = random.uniform(-2.09 , 2.09)
j_a6 = random.uniform(-6.11, 6.11)