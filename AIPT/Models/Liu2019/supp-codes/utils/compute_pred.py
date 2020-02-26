import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt,pandas as pd
from os import makedirs
import numpy as np,sys,h5py,cPickle,argparse,subprocess
from os.path import join,dirname,basename,exists
from os import system
import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import model_from_json
from keras import backend as K
import theano


arch='best_archit.json'
weight='bestmodel_weights.h5'
amino=np.asarray(['I', 'L', 'V', 'F', 'M', 'C', 'A', 'G', 'P', 'T', 'S', 'Y', 'W', 'Q', 'N', 'H', 'E', 'D', 'K', 'R', 'X'])
basedir='../data/'
nets=['Easy_classification_0622_reg','Easy_classification_0604_holdouttop_reg','Easy_classification_0615_holdouttop_reg','Easy_classification_0604_holdouttop']
models=['seq_32x1_16','seq_64x1_16','seq_32x2_16','seq_32_32','seq_32x1_16_filt3','seq_emb_32x1_16']

def main():
    	if len(sys.argv) < 2:
    	    print "you must call program as: python get_weights.py <rootpath> <your_archit.json><resultdir><l2><l1><step> "
    	    sys.exit(1)
    	resultdir=sys.argv[1]
    	if not exists(resultdir):
    	    makedirs(resultdir)
	inputdir=sys.argv[2]
	input_all=np.asarray([]).reshape((0,20, 1, 20))
        files = subprocess.check_output('ls '+join(inputdir,'data.h5.batch*'), shell=True).split('\n')[:-1]
        for batchfile in files:
                fi = h5py.File(batchfile, 'r')
                dataset = np.asarray(fi['data'])
                input_all=np.append(input_all,dataset,axis=0)
	for net in nets:
		for model in models:
			if net=='Easy_classification_0604_holdouttop':
				isreg=False
			else:
				isreg=True
			if not exists(join(basedir,net,'Lucentis_b','CV0',model)):
				continue
			name=net+'.'+model+'.score'
			print name
		#------parsing model layers----------
			outname=join(resultdir,name)
			architecture_file=join(basedir,net,'Lucentis_b','CV0',model,model+'_'+arch)
			if not exists(architecture_file):
				print 'file dont exist!skip'
				continue
			weight_file=join(basedir,net,'Lucentis_b','CV0',model,model+'_'+weight)
			model = model_from_json(open(architecture_file).read())
			model.load_weights(weight_file)
			layer_input = model.layers[0].input
			layer_size=model.layers[0].input_shape
			if isreg:
				output_layer = model.layers[-1].output
			else:   
				output_layer = model.layers[-1].output
		#--------parse input file-----------
			activation_all=output_layer[:,0]
			predict=np.asarray([])
			seq2=np.asarray([])
			iterate2= K.function([layer_input,K.learning_phase()],activation_all)
			for batch in range(0,input_all.shape[0],10000):
			    input_data=input_all[batch:min(batch+10000,input_all.shape[0])]
			    activation=iterate2([input_data,0])
			    datain=np.copy(input_data)
			    predict=np.append(predict,activation,axis=0)
			    trim1=np.sum(datain.reshape(datain.shape[0],datain.shape[1],datain.shape[3]),axis=1)!=0
			    trim2=np.tile(~trim1,(1,20)).reshape(input_data.shape[0],input_data.shape[1],input_data.shape[2],input_data.shape[3])
			    orig_seq=[]
			    trims=np.bool_(trim1)
			    onehot=datain.reshape(datain.shape[0],datain.shape[1],datain.shape[3])
			    colrank=np.argsort(onehot,axis=1)
			    colind=np.argmax(onehot,axis=1)
			    for i in range(colind.shape[0]):
				temp=colind[i][trims[i]]
				temp[onehot[i][0][trims[i]]==0.05]=20
				sequence=''.join(amino[temp])
				orig_seq.append(sequence)
			    orig_seq=np.asarray(orig_seq)
			    seq2=np.append(seq2,orig_seq,axis=0)
			tsv=np.transpose(np.asarray([seq2,predict]))
			with open(outname,'a+') as outfile: 
				np.savetxt(outfile,tsv,fmt='%s',delimiter='\t')
	score_file=join(resultdir,'scores.csv')
	score_dir=resultdir
	i=0
	for net in nets:
	    for model in models:
		pred_file = join(score_dir,'{}.{}.score'.format(net,model))        
		if not exists(pred_file):
		    continue
		score=pd.read_csv(pred_file, index_col=0,header=None,sep='\t')
		score.columns=['{}.{}.score'.format(net,model)]
		if i==0:
		    all_score=score
		else:
		    all_score=pd.concat([all_score,score],axis=1)
		i+=1
	all_score.to_csv(score_file)

if __name__ == "__main__":
        main()

