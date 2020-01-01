import torch
import torch.nn as nn
import torch.optim as optim
import loader
import os
import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, accuracy_score
import json

class Model(nn.Module):

  def __init__(self, epoch=50, batch_size=16, learning_rate=0.01, *args, **kwargs):
    super(Model, self).__init__()

    param_dict = {'work':'Model','epoch':epoch,'batch_size':batch_size,'learning_rate':learning_rate}
    if self.load_param() is None:
      self.save_param

    self.epoch = epoch
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    # self.model = None

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
          features = features.view(features.shape[0], 20*10)
          logps = self.forward(features)
          loss = self.objective()
          loss = loss(logps,labels.type(torch.long))
          outputs_train.append(logps.detach().numpy())
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      print('Epoch: %d: Loss=%.3f'%(e+1,loss))
    self.save_model('temp',self.state_dict(),os.getcwd())

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

  def save_model(self, model_name, model, path):

    modelpath = os.path.join(path,model_name)
    if not os.path.exists(modelpath):
      os.mkdir(modelpath)
      torch.save(model,os.path.join(modelpath,model_name))
    else:
      print('Model already exists')

  def load_model(self, model_name, path):

    modelpath = os.path.join(path,model_name)
    if os.path.exists(modelpath):
      self.net_init()
      return self.load_state_dict(torch.load(os.path.join(modelpath, model_name)))
    return None

  def save_param(self, path, param_dict):
    with open(os.path.join(path, '/train_parameters.json'), 'w') as f:
      json.dump(param_dict, f, indent=2)

  def load_param(self, path):
    if os.path.exists(os.path.join(path,'train_parameters.json')):
      return json.load(open(os.path.join(path,'train_parameters.json'), 'r'))
    return None

if __name__ == '__main__':

  data, out = loader.synthetic_data(num_samples=1000, seq_len=10)
  train_loader, test_loader = loader.train_test_loader(data, out, test_size=0.3, batch_size=300)

  model = Model(batch_size=15)
  if model.load_model('temp', os.getcwd()) == None:
    model.fit(train_loader)
  else:
    m = model.load_model('temp', os.getcwd())

  out_test, labels_test = model.predict(test_loader)
  mat, acc, mcc = model.evaluate(out_test, labels_test)
