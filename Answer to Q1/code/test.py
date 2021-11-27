import numpy as np
import argparse
import os
import sys
import time

from model import Model

import torch
import torch.utils.data
import torch.nn.functional as F

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(num_classes=FLAGS.num_classes, num_instances=FLAGS.num_instances, num_features=FLAGS.num_features, num_bins=FLAGS.num_bins, sigma=FLAGS.sigma)
model.to(device)

model.eval()
with torch.no_grad():

	for i in range(num_slides):
		dataset.next_slide()

		slide_id = slide_ids_arr[i]
		print('Slide {}/{}: {}'.format(i+1,num_slides,slide_id))

		slide_data_folder_path = '{}/{}'.format(data_folder_path,slide_id)
		if not os.path.exists(slide_data_folder_path):
			os.makedirs(slide_data_folder_path)

		test_metrics_filename = '{}/bag_predictions_{}.txt'.format(slide_data_folder_path,slide_id)

		# write model parameters into metrics file
		with open(test_metrics_filename,'w') as f_metrics_file:
			f_metrics_file.write('# Model parameters:\n')

			for key in FLAGS_dict.keys():
				f_metrics_file.write('# {} = {}\n'.format(key, FLAGS_dict[key]))

			f_metrics_file.write("# num_slides: {}\n".format(num_slides))

			f_metrics_file.write('# bag_id\ttruth\tpred\n')


		bag_id = 0 
		for images, targets in data_loader:
			images = images.to(device)
			# print(images.size())
			# print(targets.size())
			
			# get logits from model
			batch_logits = model(images)
			batch_probs = batch_logits
			batch_probs_arr = batch_probs.cpu().numpy()

			num_samples = targets.size(0)
			# print('num_samples: {}'.format(num_samples))

			batch_truths = np.asarray(targets.numpy(),dtype=np.float32)

			with open(test_metrics_filename,'a') as f_metrics_file:
				for b in range(num_samples):
					f_metrics_file.write('{}_{}\t'.format(slide_id,bag_id))
					f_metrics_file.write('{:.3f}\t'.format(batch_truths[b,0]))
					f_metrics_file.write('{:.3f}\n'.format(batch_probs_arr[b,0]))

					bag_id += 1




