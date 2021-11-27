import numpy as np
import argparse
from datetime import datetime
import os
import sys
import time
from model import Model
from dataset import Dataset
import torch
import torch.utils.data
from tqdm import tqdm
#############################################################################
# construct model
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(num_classes=FLAGS.num_classes, num_instances=FLAGS.num_instances, num_features=FLAGS.num_features, num_bins=FLAGS.num_bins, sigma=FLAGS.sigma)
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=FLAGS.learning_rate, weight_decay=0.0005)

# initialize weights from a file
if FLAGS.init_model_file:
	if os.path.isfile(FLAGS.init_model_file):
		state_dict = torch.load(FLAGS.init_model_file, map_location=device)
		model.load_state_dict(state_dict['model_state_dict'])
		optimizer.load_state_dict(state_dict['optimizer_state_dict'])
		print('weights loaded successfully!!!\n{}'.format(FLAGS.init_model_file))


# print model parameters
print('# Model parameters:')
for key in FLAGS_dict.keys():
	print('# {} = {}'.format(key, FLAGS_dict[key]))

print("# Training Data - num_samples: {}".format(num_patients_train))
print("# Validation Data - num_samples: {}".format(num_patients_val))


# write model parameters into metrics file
with open(metrics_file,'w') as f_metrics_file:
	f_metrics_file.write('# Model parameters:\n')

	for key in FLAGS_dict.keys():
		f_metrics_file.write('# {} = {}\n'.format(key, FLAGS_dict[key]))

	f_metrics_file.write("# Training Data - num_samples: {}\n".format(num_patients_train))
	f_metrics_file.write("# Validation Data - num_samples: {}\n".format(num_patients_val))
	
	f_metrics_file.write('# epoch\ttraining_loss\tvalidation_loss\n')


# define loss criterion
criterion = torch.nn.L1Loss()
#############################################################################




#############################################################################
for epoch in range(FLAGS.num_epochs):
	print('############## EPOCH - {} ##############'.format(epoch+1))
	training_loss = 0
	validation_loss = 0

	# train for one epoch
	print('******** training ********')
		
	num_predictions = 0

	pbar = tqdm(total=len(train_data_loader))
	
	model.train()
	for images, targets in train_data_loader:
		images = images.to(device)
		targets = targets.to(device)

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		y_logits = model(images)
		loss = criterion(y_logits, targets)
		loss.backward()
		optimizer.step()

		training_loss += loss.item()*targets.size(0)

		num_predictions += targets.size(0)

		pbar.update(1)

	training_loss /= num_predictions

	pbar.close()

	# evaluate on the validation dataset
	print('******** validation ********')

	num_predictions = 0

	pbar = tqdm(total=len(val_data_loader))

	model.eval()
	with torch.no_grad():
		for images, targets in val_data_loader:
			images = images.to(device)
			targets = targets.to(device)

			# forward
			y_logits = model(images)
			loss = criterion(y_logits, targets)

			validation_loss += loss.item()*targets.size(0)

			num_predictions += targets.size(0)

			pbar.update(1)

	validation_loss /= num_predictions

	pbar.close()

	print('Epoch=%d ### training_loss=%5.3f ### validation_loss=%5.3f' % (epoch+1, training_loss, validation_loss))

	# save model
	if (epoch+1) % FLAGS.save_interval == 0:
		model_weights_filename = '{}/model_weights__{}__{}.pth'.format(FLAGS.models_dir,current_time,epoch+1)
		state_dict = {	'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict()}
		torch.save(state_dict, model_weights_filename)
		print("Model weights saved in file: ", model_weights_filename)


model_weights_filename = '{}/model_weights__{}__{}.pth'.format(FLAGS.models_dir,current_time,epoch+1)
state_dict = {	'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()}
torch.save(state_dict, model_weights_filename)
print("Model weights saved in file: ", model_weights_filename)

print('Training finished!!!')
#############################################################################
