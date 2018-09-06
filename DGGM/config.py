# Hyper-parameters and other global variables

train = 'train_aae_10m'
test = 'test_aae_10m'

num_of_epochs = 100
batch_size_for_model = 1024
num_of_test_instances = 85399

input_layer_size, latent_layer_size, output_layer_size = 167, 20, 167
middle_layer_sizes = [256, 256]

discriminator_sizes = [64, 64, 8, 1]

logdir = 'tmp/aae/1/'

""" gpu_config:
	['NONE, -1] -> Run on GPU normally
	['GPU_MEM_FRACTION', fraction] -> Use 'fraction' percent
									of GPU and CPU otherwise.
	['ALLOW_GROWTH', -1] -> Use the 'allow_growth' protocol.
"""
gpu_config = ['GPU_MEM_FRACTION', 0.4]