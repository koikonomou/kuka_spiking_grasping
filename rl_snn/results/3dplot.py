import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

def plot_joints(data,env_num):
    px = []
    py =[]
    pz = []
    status = []
    time = []
    a = []
    b = []
    c = []
    path = data["path"]
    trials =len(data['path'][env_num])
    final_state = data["final_state"][env_num]
    for trial in range(trials):
        px.append( [path[env_num][trial][0],path[env_num][trial][3],path[env_num][trial][6],path[env_num][trial][9],path[env_num][trial][12],path[env_num][trial][15] ])
        py.append( [path[env_num][trial][1],path[env_num][trial][4],path[env_num][trial][7],path[env_num][trial][10],path[env_num][trial][13],path[env_num][trial][16] ])
        pz.append( [path[env_num][trial][2],path[env_num][trial][5],path[env_num][trial][8],path[env_num][trial][11],path[env_num][trial][14],path[env_num][trial][17] ])
    if final_state == 1:
        status.append('Success')
        time.append(data["time"][env_num])
    if final_state == 2:
        status.append("Failure")
        time.append(data["time"][env_num])
    if final_state == 3:
        status.append("Time out")
        time.append(data["time"][env_num])
    time = str(round(np.sum(time), 3))
    for y in range(len(px)):
        globals()['a%s' %y] = []
        for item in px[y]:
            globals()['a%s' %y].append(float(item))
    for y in range(len(py)):
        globals()['b%s' %y] = []
        for item in py[y]:
            globals()['b%s' %y].append(float(item))
    for y in range(len(pz)):
        globals()['c%s' %y] = []
        for item in pz[y]:
            globals()['c%s' %y].append(float(item))
    for i in range(len(px)):
        aval = np.array(globals()['a%s' %i])
        a.append(aval)
        bval = np.array(globals()['b%s' %i])
        b.append(bval)
        cval = np.array(globals()['c%s' %i])
        c.append(cval)
    return a,b,c,pz,time,status


directory = '../record_data/ddpg_simple/10_10_c'
ep_number=1
run_num = 0
file = 'episode_'+'{ep_num}'.format(ep_num=ep_number) + '_run_'+'{run}'.format(run=run_num)+'.p'
data = pickle.load(open(directory+'/'+file, 'rb'))
a,b,c,d,time,status = plot_joints(data=data,env_num=0)
fig = plt.figure()
ax = fig.add_subplot(2, 5, 1, projection='3d')
ax.set_xlabel("x axis")
ax.set_ylabel("y axis ")
ax.set_zlabel("z axis")
for i in range(len(d)):
    ax.set_title('DDPG env 1 \nexec time:%s \nstatus: %s'% (time ,status), fontsize=7)
    ax.scatter(a[i],b[i],zs = c[i], s=80)

for i in range(len(d)):
    ax.plot3D(a[i],b[i],d[i])


a,b,c,d,time,status = plot_joints(data=data,env_num=1)
ax = fig.add_subplot(2, 5, 2, projection='3d')
ax.set_xlabel("x axis")
ax.set_ylabel("y axis ")
ax.set_zlabel("z axis")
for i in range(len(d)):
    ax.set_title('DDPG env 2 \nexec time:%s \nstatus: %s'% (time ,status), fontsize=7)
    ax.scatter(a[i],b[i],zs = c[i], s=80)
for i in range(len(d)):
    ax.plot3D(a[i],b[i],d[i])

a,b,c,d,time,status = plot_joints(data=data,env_num=2)
ax = fig.add_subplot(2, 5, 3, projection='3d')
ax.set_xlabel("x axis")
ax.set_ylabel("y axis ")
ax.set_zlabel("z axis")
for i in range(len(d)):
    ax.set_title('DDPG env 3 \nexec time:%s \nstatus: %s'% (time ,status), fontsize=7)
    ax.scatter(a[i],b[i],zs = c[i], s=80)

for i in range(len(d)):
    ax.plot3D(a[i],b[i],d[i])

a,b,c,d,time,status = plot_joints(data=data,env_num=3)
ax = fig.add_subplot(2, 5, 4, projection='3d')
ax.set_xlabel("x axis")
ax.set_ylabel("y axis ")
ax.set_zlabel("z axis")
for i in range(len(d)):
    ax.set_title('DDPG env 4 \nexec time:%s \nstatus: %s'% (time ,status), fontsize=7)
    ax.scatter(a[i],b[i],zs = c[i], s=80)

for i in range(len(d)):
    ax.plot3D(a[i],b[i],d[i])

a,b,c,d,time,status = plot_joints(data=data,env_num=4)
ax = fig.add_subplot(2, 5, 5, projection='3d')
ax.set_xlabel("x axis")
ax.set_ylabel("y axis ")
ax.set_zlabel("z axis")
for i in range(len(d)):
    ax.set_title('DDPG env 5 \nexec time:%s \nstatus: %s'% (time ,status), fontsize=7)
    ax.scatter(a[i],b[i],zs = c[i], s=80)

for i in range(len(d)):
    ax.plot3D(a[i],b[i],d[i])


# SNN


directory = '../record_data/snn_without_subgoal/5_10_b'
ep_number=1
run_num = 0
file = 'episode_'+'{ep_num}'.format(ep_num=ep_number) + '_run_'+'{run}'.format(run=run_num)+'.p'
data = pickle.load(open(directory+'/'+file, 'rb'))
a,b,c,d,time,status = plot_joints(data=data,env_num=0)
ax = fig.add_subplot(2, 5, 6, projection='3d')
ax.set_xlabel("x axis")
ax.set_ylabel("y axis ")
ax.set_zlabel("z axis")
for i in range(len(d)):
    ax.set_title('SNN env 1 \nexec time:%s \nstatus: %s'% (time ,status), fontsize=7)
    ax.scatter(a[i],b[i],zs = c[i], s=80)

for i in range(len(d)):
    ax.plot3D(a[i],b[i],d[i])


a,b,c,d,time,status = plot_joints(data=data,env_num=1)
ax = fig.add_subplot(2, 5, 7, projection='3d')
ax.set_xlabel("x axis")
ax.set_ylabel("y axis ")
ax.set_zlabel("z axis")
for i in range(len(d)):
    ax.set_title('SNN env 2 \nexec time:%s \nstatus: %s'% (time ,status), fontsize=7)
    ax.scatter(a[i],b[i],zs = c[i], s=80)
for i in range(len(d)):
    ax.plot3D(a[i],b[i],d[i])

a,b,c,d,time,status = plot_joints(data=data,env_num=2)
ax = fig.add_subplot(2, 5, 8, projection='3d')
ax.set_xlabel("x axis")
ax.set_ylabel("y axis ")
ax.set_zlabel("z axis")
for i in range(len(d)):
    ax.set_title('SNN env 3 \nexec time:%s \nstatus: %s'% (time ,status), fontsize=7)
    ax.scatter(a[i],b[i],zs = c[i], s=80)

for i in range(len(d)):
    ax.plot3D(a[i],b[i],d[i])

a,b,c,d,time,status = plot_joints(data=data,env_num=3)
ax = fig.add_subplot(2, 5, 9, projection='3d')
ax.set_xlabel("x axis")
ax.set_ylabel("y axis ")
ax.set_zlabel("z axis")
for i in range(len(d)):
    ax.set_title('SNN env 4 \nexec time:%s \nstatus: %s'% (time ,status), fontsize=7)
    ax.scatter(a[i],b[i],zs = c[i], s=80)

for i in range(len(d)):
    ax.plot3D(a[i],b[i],d[i])

a,b,c,d,time,status = plot_joints(data=data,env_num=4)
ax = fig.add_subplot(2, 5, 10, projection='3d')
ax.set_xlabel("x axis")
ax.set_ylabel("y axis ")
ax.set_zlabel("z axis")
for i in range(len(d)):
    ax.set_title('SNN env 5 \nexec time:%s \nstatus: %s'% (time ,status), fontsize=7)
    ax.scatter(a[i],b[i],zs = c[i], s=80)

for i in range(len(d)):
    ax.plot3D(a[i],b[i],d[i])
plt.show()