from pyspark import SparkContext, SparkConf,AccumulatorParam
import sys
import numpy as np
import copy

class MatrixAccumulatorParam(AccumulatorParam):
    def zero(self, initialValue):
		[W,H] = initialValue
		return [W,H]

    def addInPlace(self, v1, v2):
        v1[0] += v2[0]
        v2[1] += v2[1]
        return v1

class ProcessData(object):
	def __init__(self,factors,workers,iteration,\
		b_value,l_value,input_file_name,\
		output_w_file_name,output_h_file_name):
		self.num_factors = factors
		self.num_workers = workers
		self.num_iteration = iteration
		self.beta_value = b_value
		self.lambda_value = l_value
		self.input_file_name = input_file_name
		self.output_w_file_name  = output_w_file_name
		self.output_h_file_name = output_h_file_name
		self.maxrow = 500
		self.maxcol = 500
		self.blockRow = self.maxrow/self.num_workers
		self.blockCol = self.maxcol/self.num_workers

def groupkeysByRow(triples):
	return (triples[0]-1)/pdata.blockRow

def groupkeysByStratum(triples):
	row = (triples[1][0] - 1) / pdata.blockRow
	col = (triples[1][1] - 1)/ pdata.blockCol
	return (row + col) % pdata.num_workers
		

def initializeWH():
	W = np.random.random_integers(5, size = (pdata.maxrow,pdata.num_factors))
	H = np.random.random_integers(5, size = (pdata.num_factors,pdata.maxcol))
	return [W,H]

def findmaxrowcol():
	data = np.genfromtxt(pdata.input_file_name,skiprows=n) 
	pdata.maxrow = np.argmax(np.max(data, axis=1))
	pdata.maxcol = np.argmax(np.max(data, axis=0))

def performSGD(pdata):
	trainingData = sc.textFile(pdata.input_file_name,pdata.num_workers).map(lambda x: [int(z) for z in x.split(',')])
	
	Nistar = trainingData.keyBy(lambda y: y[0]).countByKey()
	Njstar = trainingData.keyBy(lambda z: z[1]).countByKey()

	trainingData = trainingData.keyBy(groupkeysByRow)
	trainingData = trainingData.partitionBy(pdata.num_workers,lambda c:c)

	mb = sc.accumulator(0)
	sc.broadcast(Nistar)
	sc.broadcast(Njstar)
	WH = initializeWH()
	WH = sc.accumulator(WH,MatrixAccumulatorParam())


	def SGDtoBlock(keyval,WH,mb):
		W = WH[0]
		H = WH[1]
		previousW = copy.deepcopy(W)
		previousH = copy.deepcopy(H)
		n = 0

		for item in keyval[1]:
			row = item[0]
			col = item[1]
			value = item[2]
			epsilon = (100 + n + mb) ** (-pdata.beta_value)

			previousW[row - 1,:] = W[row - 1,:] - epsilon\
			 * (-2 * H[:,col - 1] * (value - W[row - 1,:]\
			  * H[:,col - 1]) + 2 * pdata.lambda_value * np.transpose(W[row - 1,:])/(Nistar or 1))

			previousH[:,col - 1] = H[:,col - 1] - epsilon\
		 	* (-2 * np.transpose(W[row - 1,:])\
		  	* (value - W[row - 1,:]*H[:,col - 1]) + 2 * pdata.lambda_value * H[:,col-1]/(Njstar or 1)) 
			n += 1

		W = previousW
		H = previousH
		return [W,H]	

	for iter in xrange(pdata.num_iteration):
		for stratum in xrange(pdata.num_workers):
			thisstratum = trainingData.filter(lambda x:x == groupkeysByStratum(x)).groupByKey()
			[W,H] = copy.deepcopy(WH.value)
			summb = copy.deepcopy(mb.value)
			thisstratum.foreach(lambda kval:WH.add(SGDtoBlock(kval,[W,H],summb)))
			thisstratum.foreach(lambda kval: summb.add(len(kval[1])))
	return WH.value


if __name__ == '__main__':
	num_factors = int(sys.argv[1])
	num_workers = int(sys.argv[2])
	num_iteration = int(sys.argv[3])
	beta_value = float(sys.argv[4])
	lambda_value = float(sys.argv[5])
	input_file_name = sys.argv[6]
	output_w_file_name = sys.argv[7]
	output_h_file_name = sys.argv[8]
	conf = SparkConf().setAppName("dsgd").setMaster("local")
	sc = SparkContext(conf=conf)
	
	pdata = ProcessData(num_factors,num_workers,num_iteration,beta_value,lambda_value,\
		input_file_name,output_w_file_name,output_h_file_name)
	
	[outputW,outputH] = performSGD(pdata)
	np.savetxt(pdata.output_w_file_name,outputW,delimiter=",")
	np.savetxt(pdata.output_h_file_name,outputH,delimiter=",")
	
