import torch
import torch.nn as nn
import torch.optim as optim
import loader
import os
import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, accuracy_score
import json

class Model(nn.Module):

  def __init__(self, model_name='Model', epoch=50, batch_size=16, learning_rate=0.01, *args, **kwargs):
    super(Model, self).__init__()

    self.model_name = model_name
    self.workpath = os.path.join(os.getcwd(),'work')
    self.modelnamepath = os.path.join(self.workpath,self.model_name)
    self.modelpath = os.path.join(self.modelnamepath,'model')

    if not os.path.exists(self.workpath):
      os.mkdir(self.workpath)
      if not os.path.exists(self.modelnamepath):
        os.mkdir(self.modelnamepath)

    if self.load_param(self.modelnamepath) is None:
      param_dict = {'model_name':model_name,'epoch':epoch,'batch_size':batch_size,'learning_rate':learning_rate}
      self.save_param(self.modelnamepath, param_dict)
      self.epoch = epoch
      self.batch_size = batch_size
      self.learning_rate = learning_rate
    else:
      param_dict = self.load_param(self.modelnamepath)
      self.epoch = param_dict['epoch']
      self.batch_size = param_dict['batch_size']
      self.learning_rate = param_dict['learning_rate']

  def net_init(self):

    self.fc = nn.Linear(20*10, 2)

  def forward(self, input1):

    input1 = input1.view(input1.shape[0], 20*10)
    input1 = self.fc(input1)
    input1 = torch.sigmoid(input1)

    return input1

  def objective(self):

      return nn.CrossEntropyLoss()

  def fit(self, train_loader):

    self.net_init()
    print()
    self.train()
    optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
    for e in range(self.epoch):
      for features, labels in train_loader:
          outputs_train = []
          # features = features.view(features.shape[0], 20*10)
          logps = self.forward(features)
          loss = self.objective()
          loss = loss(logps,labels.type(torch.long))
          outputs_train.append(logps.detach().numpy())
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          self.save_model('Epoch_'+str(e+1),self.state_dict())

      print('Epoch: %d: Loss=%.3f'%(e+1,loss))
    

  def predict(self, test_loader):

    self.eval()
    test_loss = 0
    outputs_test = []
    labels_test = []
    with torch.no_grad():
        for data_1 in test_loader:
            inputs, labels = data_1
            outputs = self.forward(inputs)
            # outputs = outputs.reshape(1,num_classes)
            batch_loss = self.objective()
            batch_loss = batch_loss(outputs, labels.type(torch.long))
            test_loss += batch_loss.item()
            outputs_test.append(outputs.detach().numpy())
            labels_test.append(labels)

        return outputs_test, labels_test

  def evaluate(self, outputs, labels):

    y_pred = []
    for a in outputs:
      y_pred.append(a[0][1]/(a[0][1]+a[0][0]))
    y_true = np.array(labels)
    y_pred = np.array(y_pred).round()
    mat = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    print('Test: ')
    print(mat)
    print('Accuracy = %.3f ,MCC = %.3f' %(acc,mcc))

    return mat, acc, mcc

  def save_model(self, filename, model):

    if not os.path.exists(self.modelpath):
      os.mkdir(self.modelpath)
      torch.save(model,os.path.join(self.modelpath,filename))
    else:
      torch.save(model,os.path.join(self.modelpath,filename))

  def load_model(self, filename):

    if os.path.exists(self.modelpath):
      self.net_init()
      return self.load_state_dict(torch.load(os.path.join(self.modelpath,filename)))
    return None

  def save_param(self, path, param_dict):
    with open(os.path.join(path, 'train_parameters.json'), 'w') as f:
      json.dump(param_dict, f, indent=2)

  def load_param(self, path):
    if os.path.exists(os.path.join(path,'train_parameters.json')):
      return json.load(open(os.path.join(path,'train_parameters.json'), 'r'))
    return None

if __name__ == '__main__':

  data, out = loader.synthetic_data(num_samples=1000, seq_len=10)
  data = loader.encode_data(data)
  train_loader, test_loader = loader.train_test_loader(data, out, test_size=0.3, batch_size=300)

  model = Model(batch_size=15)
  if model.load_model('Epoch_'+str(model.epoch)) == None:
    model.fit(train_loader)
  else:
    m = model.load_model('Epoch_'+str(model.epoch))

  out_test, labels_test = model.predict(test_loader)
  mat, acc, mcc = model.evaluate(out_test, labels_test)
