import h5py
from os.path import join,exists
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.optimizers import Adadelta,RMSprop
from hyperas.distributions import choice, uniform, conditional
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
from random import randint
from sklearn.cross_validation import train_test_split
from keras import backend as K
K.set_image_dim_ordering('th')

def reportAcc(acc,score,bestaccfile):
    print('Hyperas:valid accuracy:', acc,'valid loss',score)
    if not exists(bestaccfile):
        current = float("inf")
    else:
        with open(bestaccfile) as f:
            current = float(f.readline().strip())
    if score < current:
        with open(bestaccfile,'w') as f:
            f.write('%f\n' % score)
            f.write('%f\n' % acc)

def model(X_train, Y_train, X_test, Y_test):
    W_maxnorm = 3
    DROPOUT = {{choice([0.3,0.5,0.7])}}

    model = Sequential()
    model.add(Flatten(input_shape=(20, 1, DATASIZE)))
    model.add(Dense(32,activation='relu',W_constraint=maxnorm(W_maxnorm)))
    model.add(Dropout(DROPOUT))
    model.add(Dense(32,activation='relu',W_constraint=maxnorm(W_maxnorm)))
    model.add(Dropout(DROPOUT))
    #model.add(Dense(2,W_constraint=maxnorm(W_maxnorm)))
    #model.add(Activation('softmax'))
    model.add(Dense(1))

    myoptimizer = RMSprop(lr={{choice([1e-1,0.01,0.001,0.0001,1e-5])}}, rho=0.9, epsilon=1e-06)
    #mylossfunc = 'binary_crossentropy'
    mylossfunc='mean_squared_error'#, optimizer=myoptimizer,metrics=['accuracy'])
    model.compile(loss=mylossfunc, optimizer=myoptimizer,metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=100, nb_epoch=5,validation_split=0.1)

    score, acc = model.evaluate(X_test,Y_test)
    model_arch = 'MODEL_ARCH'
    bestaccfile = join('TOPDIR',model_arch,model_arch+'_hyperbestacc')
    reportAcc(acc,score,bestaccfile)

    return {'loss': score, 'status': STATUS_OK,'model':(model.to_json(),myoptimizer,mylossfunc)}

def data():
    myprefix = join('TOPDIR','DATACODE' + 'PREFIX')
    X_train,Y_train = getdata(myprefix + '.train.h5.batch')
    X_test,Y_test = getdata(myprefix + '.valid.h5.batch')
    return X_train, Y_train, X_test, Y_test

def BatchGenerator(batchnum,cls,topdir,data_code):
    data1prefix = join(topdir,data_code + 'PREFIX' +'.'+cls+'.h5.batch')
    for i in range(batchnum):
        data1f = h5py.File(data1prefix+str(i+1),'r')
        data1 = data1f['data']
        label = data1f['label']
        yield (data1,label)

def getdata(data1prefix):
    data1f = h5py.File(data1prefix+'1','r')
    return (data1f['data'],data1f['label'])

def BatchGenerator2(minibatch_size,batchnum,cls,topdir,data_code):
    data1prefix = join(topdir,data_code + 'PREFIX' +'.'+cls+'.h5.batch')
    leftover = 0
    while True:
        for i in range(batchnum):
            data1f = h5py.File(data1prefix+str(i+1),'r')
            data1 = data1f['data']
            label = data1f['label']
            datalen = len(data1)
            idx = 0
            if leftover >0:
                idx += minibatch_size - leftover
                yield (np.vstack((leftover_data['data'],data1[:idx])),np.vstack((leftover_data['label'],label[:idx])))
            while idx+minibatch_size <= datalen:
                idx += minibatch_size
                yield (data1[(idx-minibatch_size):idx], label[(idx-minibatch_size):idx])
            leftover = datalen - idx
            if leftover >0:
                leftover_data = {'data':data1[idx:], 'label':label[idx:]}
                if i==batchnum-1:
                    leftover = 0
                    yield leftover_data
