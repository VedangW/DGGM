#!usr/bin/python

import sys
import aae_class

def train_aae(epochs, summary_file):
	aae = aae_class.AAE(batch_size=1024, num_epochs=epochs)
	with open('./fpt.aae.log', 'w') as log:
		aae = aae.train(log, summary_file)

def main():
	train_aae(int(sys.argv[1]), sys.argv[2])

if __name__ == "__main__":
	main()