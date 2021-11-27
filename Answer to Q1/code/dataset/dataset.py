#!/usr/bin/python
#coding=utf-8

#download modules
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.datasets import mnist
import matplotlib.pyplot as plt
import random 
print('loading module has finished')

#############################################################################
#download mnist dataset
(x_train_image, y_train_label),(x_test_image, y_test_label)=mnist.load_data()
# print('train image', x_train_image.shape)
# print('test images:', x_test_image.shape)
# print('train labels:', y_train_label.shape)
# print('test labels:', y_test_label.shape)
#############################################################################



#############################################################################
#image visualization
# def plot_images_labels(images, labels, idx, num=10):
# 	fig=plt.gcf()
# 	fig.set_size_inches(8,2)
# 	if num>10:
# 		num=10
# 	for i in range(0,num):
# 		ax=plt.subplot(1,10,1+i)
# 		ax.imshow(images[idx], cmap='binary')
# 		title='label='+str(labels[idx])
# 		ax.set_title(title,fontsize=10)
# 		ax.set_xticks([])
# 		ax.set_yticks([])
# 		idx+=1
# 	plt.show()
# plot_images_labels(x_train_image,y_train_label,0,10)
#############################################################################



#############################################################################
#dataset preprocess
train_filter = np.where((y_train_label == 0 ) | (y_train_label == 7))
test_filter = np.where((y_test_label == 0) | (y_test_label == 7))
X_train, y_train = x_train_image[train_filter], y_train_label[train_filter]
X_test, y_test = x_test_image[test_filter], y_test_label[test_filter]
# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)
x_train=X_train.reshape(12188,784).astype('float')
x_test=X_test.reshape(2008,784).astype('float')
# y_trainOneHot=np_utils.to_categorical(y_train_label)
# y_testOneHOt=np_utils.to_categorical(y_test_label)
# print('reshaped train set:',x_train.shape)
# print('reshaped test set:',x_test.shape)
# print('train OneHot:',y_trainOneHot.shape)
# print('train OneHot:',y_testOneHOt.shape)
#############################################################################



#############################################################################
# preprocessed dataset visualization
def plot_images_labels(images, labels, idx, num=10):
	fig=plt.gcf()
	fig.set_size_inches(8,2)
	if num>10:
		num=10
	for i in range(0,num):
		ax=plt.subplot(1,10,1+i)
		ax.imshow(images[idx], cmap='binary')
		title='label='+str(labels[idx])
		ax.set_title(title,fontsize=10)
		ax.set_xticks([])
		ax.set_yticks([])
		idx+=1
	plt.show()
plot_images_labels(X_train,y_train,0,10)
#############################################################################



#############################################################################
#generate training set, validation set 
# np.savetxt("x_train",x_train,fmt='%.00f')
fold_0=x_train[:9000,:]
# print(fold_0.shape)
fold_1=x_train[9001:12188,:]
# print(fold_1.shape)
# np.savetxt("train set",fold_0,fmt='%.00f')
# np.savetxt("validation set",fold_1,fmt='%.00f')
# np.savetxt("test set",x_test,fmt='%.00f')
# np.savetxt("train set labels",y_train,fmt='%.00f')
# np.savetxt("test set labels",y_test,fmt='%.00f')
#############################################################################



#############################################################################
#define a function to generate bags and bag-level labels
def bag(top_index,bottom_index):
	L1=random.sample(range(top_index,bottom_index),100)
	label_0=float(0)
	num_0=0
	num_7=0
	j=1
	bag=np.array([])


	for i in L1:
		#generate 100x784 feature matrix 
		image=fold_0[i,:]
		bag=np.append(bag,image)
		bag=bag.reshape(j,784).astype('float')

		#generate bag-level label
		val=y_train[i]
		if val==label_0:
			num_0+=1
		else:
			num_7+=1
		j+=1

	bag_fraction=num_0/(num_0+num_7)
	# print(bag_fraction)

	return bag, bag_fraction
#############################################################################



#############################################################################
#define bag generating function
def train_bag():
	train_bag,train_fraction=bag(0,8999)
	return train_bag,train_fraction

def validation_bag():
	vali_bag,vali_fraction=bag(9000,12187)
	return vali_bag,vali_fraction

def test_bag():
	test_bag,test_fraction=bag(0,2007)
	return test_bag,test_fraction

#############################################################################




#############################################################################
#bag generating example
exa_bag,exa_fraction=train_bag()
np.savetxt("example bag",exa_bag,fmt='%.00f')
print(exa_bag.shape)
print(exa_fraction)
#############################################################################

