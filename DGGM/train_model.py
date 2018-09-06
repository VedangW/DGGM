#!usr/bin/python

""" File which instantiates the model and trains it
	on the dataset.
"""

import sys
import config
import aae_class

summary_file = config.logdir
num_of_epochs = config.num_of_epochs

def train_aae():
	aae = aae_class.AAE()
	with open('./fpt.aae.log', 'w') as log:
		aae = aae.train(log, summary_file, num_of_epochs)

def main():
	train_aae()

if __name__ == "__main__":
	main()