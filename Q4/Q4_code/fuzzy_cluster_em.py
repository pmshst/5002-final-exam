import numpy as np

# Load data
rows = np.loadtxt(open("Q4_Data.csv", "rb"), delimiter=",", skiprows=1, usecols= range(0, 6))

# Initialize Clustering Centers
c1 = np.matrix("1,1,1,1,1,1")
c2 = np.matrix("0,0,0,0,0,0")

# Iteration
for i in range(100):
	# Calaulate square distance to c1 & c2
	dist_1 = np.sum(np.square(rows - c1), axis=1)
	dist_2 = np.sum(np.square(rows - c2), axis=1)
	# Calculate w_c1 & w_c2
	w_c1 = dist_2/(dist_1 + dist_2)
	w_c2 = 1 - w_c1
	# Calculate SSE(sum of squared error)
	SSE = np.sum(np.multiply(dist_1, w_c1)) + np.sum(np.multiply(dist_2, w_c2))
	# Save as old value
	c1_old = c1
	c2_old = c2
	# Calculate new Clustering Centers
	c1 = np.matmul(np.transpose(np.square(w_c1)), rows) / np.sum(np.square(w_c1))
	c2 = np.matmul(np.transpose(np.square(w_c2)), rows) / np.sum(np.square(w_c2))
	# Print
	if i<=1:
		print("After iteration "+ str(i+1) + ":")
		print("the updated SSE(sum of squared error) is:")
		print(SSE)		
		print("the updated c1 is:")
		print(c1)
		print("the updated c2 is:")
		print(c2)
	# Calculate the sum of L1 distance of two clustering centers
	L1_sum = np.sum(np.absolute(c1_old - c1)) + np.sum(np.absolute(c2_old - c2))
	# Terminate
	if L1_sum <= 0.0001:
		print("After "+ str(i+1) + " iterations, with L1_sum="+str(L1_sum)+",the clusters converge.")
		break
# Print result
print("Converged c1:")
print(c1)
print("Converged c2:")
print(c2)


