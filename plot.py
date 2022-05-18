# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv('input/iris.csv', names=names)


dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.savefig('output/boxplot.pdf', dpi=1200)
dataset.hist()
pyplot.savefig('output/histogram.pdf', dpi=1200)
scatter_matrix(dataset)
pyplot.savefig('output/scatter.pdf', dpi=1200)
pyplot.interactive(True)
pyplot.show()