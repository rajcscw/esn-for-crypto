import numpy as np
from sklearn import preprocessing as pp
from reservoir import Utility, classicESN as res, ActivationFunctions as activation, ReservoirTopology as topology
import pickle
import sys
import json
from scipy import misc
from timeit import default_timer as timer

def get_bytes_as_integers_from_file(filename):
    bytes_read = open(filename, "rb").read()
    return list(bytes_read)

def get_bytes_from_image(file):
    img=misc.imread(file, mode="L")
    width = img.shape[0]
    height = img.shape[1]
    integers = list(img.tobytes())
    return integers, width, height

# Command line arguments
file_name = sys.argv[1]
key = sys.argv[2]
process_like_image = int(sys.argv[3])
encrypted_folder = sys.argv[4]

if process_like_image == 1:
    raw_integers, width, height = get_bytes_from_image(file_name)
else:
    raw_integers = get_bytes_as_integers_from_file(file_name)

# Key settings
keys = json.load(open(key))
seed = keys["random_seed"]
np.random.seed(seed)

# Append start of file
start_marker = 255
full_raw_integers = [start_marker] + raw_integers
full_raw_integers = np.array(full_raw_integers).reshape(len(full_raw_integers), 1)
size_of_file = len(raw_integers)

# Chunk size and other settings
n_values = 256 # Since we are dealing with byte (0-255)
chunk_size = keys["chunk_size"]
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
processed_count = 0
predicted_count = 0
reservoir_output_weights = []

# Hot encode
categories = list(range(n_values))
encoder = pp.OneHotEncoder(categories=[categories])
full_hot_encoded = encoder.fit_transform(full_raw_integers).toarray()

# Train reservoirs/Encrypt (Sender side)
for i in range(reservoirs_count):
    # Hot Encode
    hot_encoded = full_hot_encoded[processed_count:processed_count+chunk_size, :]

    # Form features and targets
    features, targets = Utility.formFeatureVectorsWithBias(hot_encoded)

    # Train
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
    esn.trainReservoirUsingPseudo()

    reservoir_output_weights.append(esn.outputWeight)

    processed_count += chunk_size
    processed_count = processed_count - 1

reservoir_output_weights = np.array(reservoir_output_weights).reshape((reservoirs_count, n_values, reservoir_size))

np.save(f"{encrypted_folder}/reservoir_output_weights", reservoir_output_weights)
np.save(f"{encrypted_folder}/file_size", np.array(size_of_file))
np.save(f"{encrypted_folder}/file_name", np.array(file_name.split("/")[-1]))

if process_like_image == 1:
    np.save(f"{encrypted_folder}/image_width", np.array(width))
    np.save(f"{encrypted_folder}/image_height", np.array(height))
