import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt,pandas as pd
from os import makedirs
import numpy as np,sys,h5py,cPickle,argparse,subprocess
from os.path import join,dirname,basename,exists
from os import system
import os
os.environ['KERAS_BACKEND'] = 'theano'

IsAnneal=True
avoidnc=True
buff_range=10
arch='best_archit.json'
weight='bestmodel_weights.h5'
constraint='oh'
saveh5=False
amino=np.asarray(['I', 'L', 'V', 'F', 'M', 'C', 'A', 'G', 'P', 'T', 'S', 'Y', 'W', 'Q', 'N', 'H', 'E', 'D', 'K', 'R', 'X'])

def mat2oh(datain,trim1,):
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
    return orig_seq

def main():
    	if len(sys.argv) < 4:
    	    print "you must call program as: python single_opt.py <rootpath><architecture><resultdir><k><stepsize><seed dir><task type>"
    	    sys.exit(1)
    	base = sys.argv[1]
    	arc=sys.argv[2]
    	architecture_file=join(base,arc+arch)
    	weight_file=join(base,arc+weight)
    	resultdir=sys.argv[3]
	iteration=int(sys.argv[4])
    	step=float(sys.argv[5])
    	if not exists(resultdir):
    	    system('mkdir '+resultdir)
	inputdir=sys.argv[6]
	type=sys.argv[7]
	savename=join(resultdir,'input_gen'+str(iteration)+'-'+str(step))
        outname=join(resultdir,'genseq-'+str(iteration)+'-'+str(step)+'.tsv')
	if exists(outname):
		print "exist!skip"
		return
	from keras.models import model_from_json
	from keras import backend as K
	import theano
#------parsing model layers----------
    	model = model_from_json(open(architecture_file).read())
    	model.load_weights(weight_file)
    	layer_input = model.layers[0].input
    	layer_size=model.layers[0].input_shape
    	print "input size:",layer_size
	if type=='reg':
    		output_layer = model.layers[-1].output
	elif type=='class':
		output_layer = model.layers[-2].output
#---------create h5py file----------
	if saveh5:
    		f = h5py.File(savename,'w')
    		grp =f.create_group("inputgen")
    		grp2=f.create_group("track")
#--------parse input file-----------
	input_all=np.asarray([]).reshape((0,layer_size[1],layer_size[2],layer_size[3]))
	files = subprocess.check_output('ls '+join(inputdir,'data.h5.batch*'), shell=True).split('\n')[:-1]
        for batchfile in files:
                fi = h5py.File(batchfile, 'r')
                dataset = np.asarray(fi['data'])
                label=np.asarray(fi['label'])
                label=label[:,0]
                input_all=np.append(input_all,dataset,axis=0)
	if saveh5:
        	grp.create_dataset("orig_seq",data=input_all.reshape(input_all.shape[0],input_all.shape[1],input_all.shape[3]))	
#---------optimization--------------
   	best_map=np.asarray([]).reshape((0,layer_size[1],layer_size[2],layer_size[3]))
	record_all=np.asarray([]).reshape((0,layer_size[1],layer_size[2],layer_size[3]))
	record_seed=np.asarray([])#seed for each proposed sequence
	oh_map=np.asarray([]).reshape((0,layer_size[1],layer_size[2],layer_size[3]))
        oh_act=np.asarray([])
	oh_seq=np.asarray([])
	final_act=np.asarray([])
	record_act=np.asarray([])
	record_sact=np.asarray([])
	best_act=np.asarray([])
	init_act=np.asarray([])
	init_loss=np.asarray([])
	best_it = np.asarray([])
	convg_it = np.asarray([])
	for batch in range(0,input_all.shape[0],2000):
	    input_data=input_all[batch:min(batch+2000,input_all.shape[0])]
	    datain=np.copy(input_data)	
	    tryi=input_data.shape[0]
       	    activation = K.sum(output_layer[:,0])
	    activation_all=output_layer[:,0]
	    if constraint=='l2':
    	   	loss = activation-L2coeff*K.sum((layer_input ** 2))# - L1coeff * K.sum(abs(layer_input))
	   	loss_all=activation_all-L2coeff*K.sum(K.sum(K.sum((layer_input**2),1),1),1)#ll=L1coeff*K.sum(K.sum(K.sum(K.abs(layer_input),1),1),1) 
	    elif constraint=='oh':
		loss = activation
		loss_all=activation_all
   	    grads = K.gradients(loss, layer_input)
    	    iterate = K.function([layer_input,K.learning_phase()], grads)
	    iterate2= K.function([layer_input,K.learning_phase()],[activation_all,loss_all])
	    print 'processing batch',batch
	    best_input =np.zeros(input_data.shape)
	    record_all_seq=np.asarray([]).reshape(0,datain.shape[1],datain.shape[2],datain.shape[3])
	    record_all_act=np.asarray([])
	    record_all_orig=np.asarray([])
	    record_seed_act=np.asarray([])
	    convg_input=np.zeros(input_data.shape)
	    best_activation = np.asarray([-1.0]*tryi)
	    best_loss = np.asarray([-1000000000.0]*tryi)
	    convg_activation = np.asarray([-10000.0]*tryi)
	    best_iter = np.asarray([-1]*tryi)
	    convg_iter=np.asarray([-1]*tryi)
	    loss_track = []
	    activation_track = []
	    count=0
	    activation_init,loss_init=iterate2([input_data,0])
	    init_act=np.append(init_act,activation_init,axis=0)
	    init_loss=np.append(init_loss,loss_init,axis=0)
	    holdcnt = np.asarray([0]*tryi)
	    lr=step
	    print 'initial activation,loss=',(np.mean(activation_init),np.mean(loss_init))
	    mask=np.array([False for i in range(tryi)])
	    trim1=np.sum(datain.reshape(datain.shape[0],datain.shape[1],datain.shape[3]),axis=1)!=0
	    trim2=np.tile(~trim1,(1,20)).reshape(input_data.shape[0],input_data.shape[1],input_data.shape[2],input_data.shape[3])
	    record_trim=np.asarray([]).reshape(0,trim1.shape[1])
	    orig_seq=mat2oh(datain,trim1)
	    while True:
		for grow in range(iteration):
			grads_value = iterate([input_data,0])
			grads_value[mask,:,:,:]=0
			if count>100:
                            lr=max(step*(count-100)**(-0.2),1e-6)
			input_data+= grads_value*lr
			input_data[trim2]=0.0
		activation_all,loss_all=iterate2([input_data,0])
	   	print 'Iteration',count*iteration
		print 'Activation',np.mean(activation_all)
		print 'Converged',sum(mask)
		onehot=input_data.reshape(input_data.shape[0],input_data.shape[1],input_data.shape[3])
		colrank=np.argsort(onehot,axis=1)
                colind=np.argmax(onehot,axis=1)
            	oh=np.zeros(onehot.shape)
            	for x in range(onehot.shape[0]):
            	    for y in range(onehot.shape[2]):
			if avoidnc:
			   if amino[colind[x,y]]=='N' or amino[colind[x,y]]=='C':
			   #print 'N detected, changing to second large'
			        colind[x,y]=colrank[x,-2,y]
                    	oh[x,colind[x,y],y]=1.0
		oh=oh.reshape(oh.shape[0],layer_size[1],layer_size[2],layer_size[3])
		oh[trim2]=0.0
            	oh_activation,oh_loss=iterate2([oh,0])
		print 'One hot Activation',np.mean(oh_activation)
		activation_track.append(oh_activation)
		temp_act=np.copy(oh_activation)
		temp_act[mask]=-10000.0
		improve=(temp_act>best_activation)
		if sum(improve)>0:
                     best_activation[improve] = oh_activation[improve]
                     best_input[improve,:,:,:] = oh[improve,:,:,:]
		     record_all_orig=np.append(record_all_orig,orig_seq[improve],axis=0)
		     record_seed_act=np.append(record_seed_act,activation_init[improve],axis=0)
		     record_all_seq=np.append(record_all_seq,oh[improve,:,:,:],axis=0)
		     record_all_act=np.append(record_all_act,oh_activation[improve],axis=0)
		     record_trim=np.append(record_trim,trim1[improve,:],axis=0)
                     best_iter[improve]=count
		holdcnt[improve]=0
		holdcnt[~improve]=holdcnt[~improve]+1
		mask=(holdcnt>=buff_range)
		if sum(mask)==tryi or count>1000:
              	     print 'Converge at',count
                     print 'Activation',np.mean(activation_all)
                     print 'Converged',sum(mask)
                     print 'Loss',np.mean(loss_all)
                     print 'Best score',np.mean(best_activation)
		     out_seq=mat2oh(record_all_seq,record_trim)
            	     oh_seq=np.append(oh_seq,out_seq,axis=0)
                     break
		count=count+1
	    best_map=np.append(best_map,best_input,axis=0)
	    best_act=np.append(best_act,best_activation,axis=0)
	    best_it=np.append(best_it,best_iter,axis=0)
	    record_all=np.append(record_all,record_all_seq,axis=0)
	    record_act=np.append(record_act,record_all_act,axis=0)
	    record_seed=np.append(record_seed,record_all_orig,axis=0)
	    record_sact=np.append(record_sact,record_seed_act,axis=0)
	if saveh5:
	    grp2.create_dataset("init_act",data=init_act)
	    grp2.create_dataset("init_loss",data=init_loss)
	    grp.create_dataset("best_seq",data=best_map)
	    grp.create_dataset("onehot_matrix",data=record_all)
	    grp2.create_dataset("original_sequence", data=orig_seq)
	    grp2.create_dataset("onehot_sequence", data=oh_seq)
	    grp2.create_dataset("seed_sequence",data=record_seed)
	    grp2.create_dataset("best_activation", data=best_act)
	    grp2.create_dataset("all_activation", data=record_act)
	    grp2.create_dataset("seed_activation", data=record_sact)
	    f.close()
	tsv=np.concatenate((np.arange(oh_seq.shape[0]).reshape(oh_seq.shape[0],1),oh_seq.reshape(oh_seq.shape[0],1),record_seed.reshape(record_seed.shape[0],1),record_act.reshape(record_act.shape[0],1),record_sact.reshape(record_sact.shape[0],1)),axis=1)
	np.savetxt(outname,tsv,fmt='%s',delimiter='\t')
	
if __name__ == "__main__":
        main()

