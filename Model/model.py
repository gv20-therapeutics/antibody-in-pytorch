import torch
import torch.nn as nn
import torch.optim as optim
import loader
import numpy as np
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, accuracy_score

class Model(nn.Module):

  def __init__(self,epoch=50,batch_size=16,learning_rate=0.001,*args, **kwargs):
    super(Model, self).__init__()

    self.epoch = epoch
    self.batch_size = batch_size
    self.learning_rate = learning_rate

  def net_init(self):

    # self.batch_size, seq_len, num_features = input.size
    self.fc = nn.Linear(20*10, 2)
    self.sigmoid = nn.Sigmoid()

  def forward(self, input):

    input = self.fc(input)
    input = self.sigmoid(input)
    return input

  def objective(self):

      return nn.CrossEntropyLoss()

  def fit(self, train_loader):

    self.net_init()
    self.train()
    optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
    for e in range(self.epoch):
      print('Epoch %d:'%(e+1))
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

      print('Loss: %.3f'%(loss))

  def predict(self, test_loader):

    self.eval()
    test_loss = 0
    outputs_test = []
    labels_test = []
    with torch.no_grad():
        for data_1 in test_loader:
            inputs, labels = data_1
            inputs = inputs.view(inputs.shape[0], 20*10)
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
    y_true = np.array(labels_test)
    y_pred = np.array(y_pred).round()
    mat = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    return mat, acc, mcc

if __name__ == '__main__':

  data, out = loader.synthetic_data(num_samples=1000, seq_len=10)
  train_loader, test_loader = loader.train_test_loader(data, out, test_size=0.3, batch_size=15)

  model = Model(batch_size=15)
  model.fit(train_loader)
  out_test, labels_test = model.predict(test_loader)
  mat, acc, mcc = model.evaluate(out_test, labels_test)

  print('Test: ')
  print(mat)
  print('Accuracy = %.3f ,MCC = %.3f' %(acc,mcc))



