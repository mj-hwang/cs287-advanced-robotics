import numpy as np 

def q_to_euler(q):
	# returns an equivalent euler angle representation of this quaternion
	# in roll-pitch-yaw order
	mData = q
	my_epsilon = 1e-10
	# euler = np.zeros((3,1))
	euler = np.zeros(3)

	# quick conversion to Euler angles to give tilt to user
	sqw = mData[3]*mData[3]
	sqx = mData[0]*mData[0]
	sqy = mData[1]*mData[1]
	sqz = mData[2]*mData[2]

	euler[1] = np.arcsin(2.0 * (mData[3]*mData[1] - mData[0]*mData[2]))
	if (np.pi/2 - np.abs(euler[1]) > my_epsilon):
		euler[2] = np.arctan2(2.0 * (mData[0]*mData[1] + mData[3]*mData[2]),sqx - sqy - sqz + sqw)
		euler[0] = np.arctan2(2.0 * (mData[3]*mData[0] + mData[1]*mData[2]),sqw - sqx - sqy + sqz)
	else:
		# compute heading from local 'down' vector
		euler[2] = np.arctan2(2*mData[1]*mData[2] - 2*mData[0]*mData[3], 2*mData[0]*mData[2] + 2*mData[1]*mData[3])
		euler[0] = 0.0
	        
		# If facing down, reverse yaw
		if (euler[1] < 0):
			euler[2] = np.pi - euler[2]
	return euler 

def axis_angle_dynamics_update(axis_angle0, pqr_times_dt):
	q0 = quaternion_from_axis_rotation(axis_angle0);
	q1 = quat_multiply(q0, quaternion_from_axis_rotation(pqr_times_dt));
	return axis_rotation_from_quaternion(q1);

def axis_rotation_from_quaternion(q):
	rotation_angle = 2 * np.arcsin(np.linalg.norm(q[:3]))
	my_eps = 1e-6
	if(rotation_angle < my_eps):
		# a = np.zeros((3,1))
		a = np.zeros(3)
	else:
		a = q[:3]/np.linalg.norm(q[:3]) * rotation_angle
	return a

def euler_to_q(euler):

	c1 = np.cos(euler[2] * 0.5)
	c2 = np.cos(euler[1] * 0.5)
	c3 = np.cos(euler[0] * 0.5)
	s1 = np.sin(euler[2] * 0.5)
	s2 = np.sin(euler[1] * 0.5)
	s3 = np.sin(euler[0] * 0.5)

	# q = np.zeros((4,1))
	q = np.zeros(4)

	q[0] = c1*c2*s3 - s1*s2*c3
	q[1] = c1*s2*c3 + s1*c2*s3
	q[2] = s1*c2*c3 - c1*s2*s3
	q[3] = c1*c2*c3 + s1*s2*s3

	return q

def express_vector_in_quat_frame(vin, q):
	# print(q[:3])
	# print(q[3])
	return rotate_vector(vin, np.append(-q[:3], q[3]))

def quaternion_from_axis_rotation(axis_rotation):
	# pass
	# 1/0
	rotation_angle = np.linalg.norm(axis_rotation)
	# quat = np.zeros((4,1))
	quat = np.zeros(4)
	#1/0
	if (rotation_angle < 1e-4):
		quat[:3] = axis_rotation/2
	else: 
		normalized_axis = axis_rotation / rotation_angle
		quat[:3] = normalized_axis * np.sin(rotation_angle / 2)
	
	quat[3] = np.sqrt(1 - np.linalg.norm(quat[:3])**2)
	return quat

def quat_multiply(lq, rq):
	# quaternion entries in order: x, y, z, w
	# quat = np.zeros((4, 1))
	quat = np.zeros(4)
	quat[0] = lq[3]*rq[0] + lq[0]*rq[3] + lq[1]*rq[2] - lq[2]*rq[1]
	quat[1] = lq[3]*rq[1] - lq[0]*rq[2] + lq[1]*rq[3] + lq[2]*rq[0]
	quat[2] = lq[3]*rq[2] + lq[0]*rq[1] - lq[1]*rq[0] + lq[2]*rq[3]
	quat[3] = lq[3]*rq[3] - lq[0]*rq[0] - lq[1]*rq[1] - lq[2]*rq[2]
	return quat

def rotate_vector(vin, q):
	# temp = quat_multiply(q, np.concatenate([vin, 0]))
	temp = quat_multiply(q, np.append(vin, 0))
	# vout = quat_multiply(temp, np.concatenate([-q[:3], q[3]]))
	vout = quat_multiply(temp, np.append(-q[:3], q[3]))
	return vout[:3]

def rotate_vector_by_inverse_quaternion(vin, q):
	vout = quat_multiply(quat_multiply(np.append(-q[:3], q[3]), np.append(vin, 0)), q)
	return vout[:3]