import kinpy as kp
import numpy as np

chain = kp.build_chain_from_urdf(open("kuka_model.urdf").read())
print(chain)
print(chain.get_joint_parameter_names())
th = [0.0, np.pi /180, 190 * np.pi / 180, 0.0, 35 * np.pi / 180, 0.0]
ret = chain.forward_kinematics(th)

print(ret)
viz = kp.Visualizer()
viz.add_robot(ret, chain.visuals_map(), axes=True)
viz.spin()