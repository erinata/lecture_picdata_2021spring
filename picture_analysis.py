import imageio
import numpy
import os
import glob
import pandas

from sklearn import preprocessing
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as pyplot

def getrgb(filepath):
	imimage = imageio.imread(filepath, pilmode='RGB')
	imimage_process = imimage/255
	imimage_process = imimage_process.sum(axis=0).sum(axis=0)/imimage_process.shape[0]*imimage_process.shape[1]
	imimage_process = imimage_process/numpy.linalg.norm(imimage_process, ord=None)
	return imimage_process


# image_one = getrgb('data/pic01.jpeg')
# print(image_one)

dataset=pandas.DataFrame()

for filepath in glob.glob('data/*'):
	image_features = pandas.DataFrame(getrgb(filepath))
	image_features = pandas.DataFrame.transpose(image_features)
	image_features['path'] = filepath
	dataset = pandas.concat([dataset, image_features])


print(dataset)

gmm_data = dataset.iloc[:,0:3]
gmm_data = preprocessing.normalize(gmm_data)
# print(gmm_data)

gmm_machine = GaussianMixture(n_components = 3)
gmm_machine.fit(gmm_data)
gmm_results = gmm_machine.predict(gmm_data)
print(gmm_results)

pyplot.scatter(dataset[0],dataset[1], c=gmm_results)
pyplot.savefig('scatter.png')
dataset['result'] = gmm_results




dataset = dataset.sort_values(by=['path'])

print(dataset)


