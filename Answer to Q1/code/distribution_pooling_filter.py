import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init

class DistributionPoolingFilter(nn.Module):
	__constants__ = ['num_bins', 'sigma']

	def __init__(self, num_bins=21, sigma=0.0167):
		super(DistributionPoolingFilter, self).__init__()

		self.num_bins = num_bins
		self.sigma = sigma
		self.alfa = 1/math.sqrt(2*math.pi*(sigma**2))
		self.beta = -1/(2*(sigma**2))

		sample_points = torch.linspace(0,1,steps=num_bins, dtype=torch.float32, requires_grad=False)
		self.register_buffer('sample_points', sample_points)


	def extra_repr(self):
		return 'num_bins={}, sigma={}'.format(
			self.num_bins, self.sigma
		)


	def forward(self, data):
		batch_size, num_instances, num_features = data.size()

		sample_points = self.sample_points.repeat(batch_size,num_instances,num_features,1)
		# sample_points.size() --> (batch_size,num_instances,num_features,num_bins)

		data = torch.reshape(data,(batch_size,num_instances,num_features,1))
		# data.size() --> (batch_size,num_instances,num_features,1)

		diff = sample_points - data.repeat(1,1,1,self.num_bins)
		diff_2 = diff**2
		# diff_2.size() --> (batch_size,num_instances,num_features,num_bins)

		result = self.alfa * torch.exp(self.beta*diff_2)
		# result.size() --> (batch_size,num_instances,num_features,num_bins)

		out_unnormalized = torch.sum(result,dim=1)
		# out_unnormalized.size() --> (batch_size,num_features,num_bins)

		norm_coeff = torch.sum(out_unnormalized, dim=2, keepdim=True)
		# norm_coeff.size() --> (batch_size,num_features,1)

		out = out_unnormalized / norm_coeff
		# out.size() --> (batch_size,num_features,num_bins)
		
		return out























