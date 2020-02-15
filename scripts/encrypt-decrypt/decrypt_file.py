import numpy as np
from sklearn import preprocessing as pp
from reservoir import Utility, classicESN as res, ActivationFunctions as activation, ReservoirTopology as topology
from datetime import datetime
import pickle
import json
import sys
from scipy import misc
from timeit import default_timer as timer

# Start timer
start = timer()

def soft_max_to_actual_values(soft_max_array):
    max = np.max(soft_max_array)
    actual = np.zeros(soft_max_array.shape)
    actual[soft_max_array == max] = 1.0
    return actual

def save_file_from_bytes(bytes, width, height, file_name):
    img_array = np.frombuffer(buffer=bytes, dtype=np.uint8)
    img_array = img_array.reshape((width, height))
    misc.imsave(file_name, img_array)

# Key settings
key = sys.argv[1]
process_like_image = int(sys.argv[2])
encrypted_folder = sys.argv[3]
output_folder = sys.argv[4]
keys = json.load(open(key))
seed = keys["random_seed"]
np.random.seed(seed)

# Get the encrypted data
reservoir_output_weights = list(np.load(f"{encrypted_folder}/reservoir_output_weights.npy"))
size_of_file = int(np.load(f"{encrypted_folder}/file_size.npy"))
file_name = np.load(f"{encrypted_folder}/file_name.npy")

# Other parameters (potential keys)
n_values = 256
start_marker = 255
chunk_size = keys["chunk_size"] # 500 bytes
reservoir_size = int(chunk_size * 0.95)
reservoirs_count = int(np.ceil(size_of_file/(chunk_size-1)))
leaking_rate = keys["leaking_rate"]
spectral_radius = keys["spectral_radius"]
input_scaling = keys["input_scaling"]
reservoir_scaling = keys["reservoir_scaling"]
initial_transient = keys["initial_transient"]
input_connectivity = keys["input_connectivity"]
reservoir_connectivity = keys["reservoir_connectivity"]
inputWeight = topology.RandomInputTopology(inputSize=n_values+1, reservoirSize=reservoir_size, inputConnectivity=input_connectivity).generateWeightMatrix(input_scaling)
reservoirWeight = topology.RandomReservoirTopology(size=reservoir_size, connectivity=reservoir_connectivity).generateWeightMatrix(reservoir_scaling)
reservoir_activation = activation.LogisticFunction()
output_activation = activation.SoftMax()
features = np.zeros((10,2)) # Dummy
targets = np.zeros((10,1)) # Dummy
predicted_count = 0
generated = []

categories = list(range(n_values))
encoder = pp.OneHotEncoder(categories=[categories])
last = encoder.fit_transform(np.array(start_marker).reshape(1,1)).toarray().reshape((1, n_values))
for output_weight in reservoir_output_weights:

    # Generate the reservoir using the pre-agreed input weight and reservoir weights (keys)
    # and output weight obtained from the sender (encrypted data)
    esn = res.Reservoir(size=reservoir_size,
                        spectralRadius=spectral_radius,
                        inputScaling=input_scaling,
                        reservoirScaling=reservoir_scaling,
                        leakingRate=leaking_rate,
                        initialTransient=initial_transient,
                        inputData=features,
                        outputData=targets,
                        reservoirActivationFunction=reservoir_activation,
                        outputActivationFunction=output_activation,
                        inputWeightRandom=inputWeight,
                        reservoirWeightRandom=reservoirWeight)
    esn.outputWeight = output_weight

    # And then predict one character at a time
    for j in range(chunk_size-1):
        if predicted_count < size_of_file:
            last = np.hstack((np.ones((last.shape[0], 1)), last)).reshape((last.shape[0], last.shape[1]+1))
            pred = soft_max_to_actual_values(esn.predictOnePoint(last))
            pred = pred.reshape(1, pred.shape[0])
            last = pred

            integer = int(np.where(last[0] == 1.0)[0][0])
            byte = integer.to_bytes(length=1, byteorder="big")
            generated.append(byte)

            predicted_count += 1

# Write it back to file
generated = b''.join(generated)
key_name = key.split("/")[-1]
if process_like_image == 1:
    width = int(np.load(f"{encrypted_folder}/image_width.npy"))
    height = int(np.load(f"{encrypted_folder}/image_height.npy"))
    save_file_from_bytes(generated, width, height, f"{output_folder}/generated_file_{key_name}_{file_name}")
else:
    output_file = open(f"{output_folder}/generated_file_{key_name}_{file_name}", 'wb')
    output_file.write(generated)