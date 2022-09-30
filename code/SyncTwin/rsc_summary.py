import numpy as np

sim_id = "sync1"
seed = 100
base_path_data = "data/{}-seed-".format(sim_id) + str(seed)
data_path = base_path_data + "/{}-{}.{}"
weight_path = base_path_data + "/{}.{}"
w = np.loadtxt(weight_path.format("w-rsc", ".csv"))

print(w.shape)
print(np.sum(w != 0))

arr = np.sum(w != 0, axis=0)
arr = arr[arr > 0]
print(np.mean(arr))
