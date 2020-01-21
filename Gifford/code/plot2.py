import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import pylab,random,cPickle
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
from os.path import dirname, basename,join,exists
from os import makedirs,system,listdir
import subprocess
from itertools import izip
import h5py
from sklearn.metrics import r2_score,mean_squared_error,roc_auc_score, accuracy_score,roc_curve,auc, precision_recall_curve,average_precision_score
from scipy.stats import ranksums,rankdata,pearsonr,spearmanr

plt.style.use('default')
plt.style.use('seaborn-white')
sns.set(color_codes=True)
sns.set_style("white")
sns.set_context("paper",font_scale=1.3)

combine_count=pd.read_csv('../data/1st_exp_counts.csv', index_col=0)#,usecols=[0,10,24,38])
combine_freq=pd.read_csv('../data/1st_exp_freq.csv', index_col=0)#,usecols=[0,10,24,38])
combine_count1=pd.read_csv('../data/1st_exp_replicate_counts.csv',index_col=0)
allreads1=combine_count1.sum(axis=0)
combine_freq1=(combine_count1+1)*100000/allreads1
antigen3=combine_freq1.columns
compressed=pd.read_pickle('../results/intermediate/all_proposed_seq/compressed_proposed_seedmap.pkl')
seedmap=pd.read_pickle('../results/intermediate/all_proposed_seq/data_proposed_seedmap.pkl')
gifford2=pd.read_csv('../data/2nd_experiment_spreadsheet.csv',index_col=0)
savedir='../results/plots'
print "plotting 1d"
fig=plt.figure(figsize=(7.5,2))
ax1 = plt.subplot(121)
plt.xlim(8,19)
ax2 = plt.subplot(122)
plt.xlim(8,19)
seq=list(combine_count[combine_count['Lucentis_a_R1']>0].index)
length=np.asarray([len(x) for x in seq])
ax=sns.distplot(length,bins=np.arange(7,21,1),kde=False,hist_kws={"normed":1,"label":"Lucentis Round 1","edgecolor":'black','alpha':1},axlabel="Lucentis Round 1",color='y',ax=ax1)
seq=list(combine_count[combine_count['Lucentis_b_R3']>0].index)
length=np.asarray([len(x) for x in seq])
ax=sns.distplot(length,bins=np.arange(7,21,1),kde=False,hist_kws={"normed":1,"label":"Lucentis Round 3","edgecolor":'black','alpha':1},axlabel="Lucentis Round 3",color='y',ax=ax2)
fig.savefig(join(savedir,'1d_cdr_length_yellow.pdf'),bbox_inches="tight")
print "plotting 1b"
th=2
sns.set_context("paper",font_scale=1.2)
plt.figure(figsize=(5.3,3))
a2='Lucentis_b_R2'
a3='Lucentis_b_R3'
allreads=combine_count.sum(axis=0)
combine_freq=(combine_count+1)*100000/allreads
chart=combine_freq[((combine_count[a2]>th)|(combine_count[a3]>th))][[a2,a3]]
rat=chart[a3]/chart[a2]
log_rat=np.log10(rat)
ax=sns.distplot(log_rat[log_rat>=0].values,bins=np.arange(0,2.5,0.1),kde=False,hist_kws={"normed":0,"label":'Strong binder',"alpha":1,"edgecolor":'black'},color='r',axlabel="log(R3/R2)")
ax=sns.distplot(log_rat[log_rat<0].values,bins=np.arange(-2.5,0.1,0.1),kde=False,hist_kws={"normed":0,"label":'Weak/Non-binder',"alpha":1,"edgecolor":'black'},color='b',axlabel="log(R3/R2)")
plt.xlim(-2.5, 2.5)
plt.legend()
plt.title(a2[0:-3])
fig = ax.get_figure()
fig.savefig(join(savedir,'1b_hist_R3R2_wkst.pdf'),bbox_inches="tight")
print "plotting 1c"
sns.set_context("paper",font_scale=1.2)
th=4
plt.figure(figsize=(4,3.4))
chart=combine_freq[((combine_count[a2]>th)|(combine_count[a3]>th))|((combine_count[a2]>0)&(combine_count[a3]>0))][[a2,a3]]
rat=chart[a3]/chart[a2]
log_rat=np.log10(rat)
chart=np.log10(chart)-5.0
plt.scatter(x=chart[a2].values,y=chart[a3].values,s=4,edgecolors='none',c=log_rat.values,vmax=2,cmap=cc.cm['diverging_gkr_60_10_c40'])
plt.xlabel('Lucentis log$_{10}$(R2)')
plt.ylabel('Lucentis log$_{10}$(R3)')
plt.colorbar()
plt.savefig(join(savedir,'1c_scatter_rdbgr.pdf'),bbox_inches="tight")
print "plotting 2d"
th2=3
th3=3
a1=antigen3[0]
a2=antigen3[1]
a3=antigen3[2]
b2='Lucentis_b_R2'
b3='Lucentis_b_R3'
cb12=combine_freq[(((combine_count[b3]>0)&(combine_count[b2]>0))|(combine_count[b3]>=th3)|(combine_count[b2]>=th2))][[b3,b2]].replace(0,0.02)
ratb12=cb12[b3]/cb12[b2]
log_ratb12=np.log10(ratb12)
c12=combine_freq1[((combine_count1[a3]>5)|(combine_count1[a2]>5))][[a3,a2]]
rat12=c12[a3]/c12[a2]
log_rat12=np.log10(rat12)
ov=list(set(log_rat12.index)&set(log_ratb12.index))
diff=log_rat12.loc[ov]-log_ratb12[ov]
uniq2=list(set(log_rat12.index)-set(log_ratb12.index))
train2=log_rat12.loc[uniq2]-diff.mean()
all_score=pd.read_csv('../data/replicate_cnn_scores.csv',index_col=0)
reg_score=all_score[all_score.columns[0:-5]]
mo=reg_score.columns
perform7=[]
for i in [0]:
    valid=uniq2
    label=train2.values
    pred=(reg_score.loc[valid]).values
    for j in range(0,len(reg_score.columns)):
        t_MSE = mean_squared_error(label,pred[:,j])
        perform7.append([t_MSE,'Test MSE',mo[j],'single network'])
        r2=r2_score(label,pred[:,j])
        perform7.append([r2,'R^2',mo[j],'single network'])
        pearson=pearsonr(label,pred[:,j])[0]
        perform7.append([pearson,'PearsonR',mo[j],'single network'])
        sprm=spearmanr(label,pred[:,j])[0]
        perform7.append([sprm,'SpearmanR',mo[j],'single network'])
    pred2=np.sort(pred,axis=1)
    for j in range(0,6):
        t_MSE = mean_squared_error(label,pred2[:,j])
        perform7.append([t_MSE,'Test MSE','lower'+str(j),'ensemble lowerbond_'+str(j)])
        r2=r2_score(label,pred2[:,j])
        perform7.append([r2,'R^2','lower'+str(j),'ensemble lowerbond_'+str(j)])
        pearson=pearsonr(label,pred2[:,j])[0]
        perform7.append([pearson,'PearsonR','lower'+str(j),'ensemble lowerbond_'+str(j)])
        sprm=spearmanr(label,pred2[:,j])[0]
        perform7.append([sprm,'SpearmanR','lower'+str(j),'ensemble lowerbond_'+str(j)])
    pred2=np.mean(pred,axis=1)
    t_MSE = mean_squared_error(label,pred2)
    perform7.append([t_MSE,'Test MSE','mean','ensemble_mean'])
    r2=r2_score(label,pred2)
    perform7.append([r2,'R^2','mean','ensemble_mean'])
    pearson=pearsonr(label,pred2)[0]
    perform7.append([pearson,'PearsonR','mean','ensemble_mean'])
    sprm=spearmanr(label,pred2)[0]
    perform7.append([sprm,'SpearmanR','mean','ensemble_mean'])
perform7pd= pd.DataFrame(perform7,columns=['Value','Metric','Model','method'])
fig,axs= plt.subplots(1,3,figsize=(9,3.5))
sns.set_context("paper",font_scale=1.3)
i=0
title=['Pearson R','R$^2$','MSE']
for me in ['PearsonR','R^2','Test MSE']:
    prsr=perform7pd[perform7pd['Metric']==me].sort_values(by='Value')
    plt.tight_layout()
    p=sns.boxplot(x='method',y='Value',data=prsr,palette=['#BBBBBB','#DDDDDD'],order=['single network','ensemble_mean'],ax=axs[i]) 
    p=sns.stripplot(x='method',y='Value',data=prsr,jitter=True,palette='Set1',order=['single network','ensemble_mean'],size=6.5,linewidth=2,ax=axs[i])
    p.set_xticklabels(['Single net','Ensemble mean'])
    p.set_ylabel(title[i]+' with log(R3/R2)')
    i+=1
fig.savefig(join(savedir,'2d_ensemble_vs_single.pdf'),bbox_inches="tight")
print "plotting 2b"
model='Easy_classification_0622_reg2.seq_32x1_16_filt3.score'
real=train2.values
pred_D=reg_score[model].loc[uniq2].values
plot=(sns.jointplot(x=real,y=pred_D,color='b',alpha=0.6,size=4,annot_kws={"loc":(-0.08,0.90),'fontsize':13},joint_kws={'s':5}).set_axis_labels("log(R3/R2)", "Predicted enrichment"))
plot.savefig(join(savedir,'2b_regression.pdf'),bbox_inches="tight")

print "plotting 2c"
thr=0.1
thr2=5
sns.set_context("paper",font_scale=1.2)
c12=combine_freq1[(((combine_count1[a3]>0)&(combine_count1[a2]>0))|(combine_count1[a3]>=th3)|(combine_count1[a2]>=th2))][[a3,a2]].replace(0,0.02)
rat12=c12[a3]/c12[a2]
log_rat12=np.log10(rat12.replace(0,0.01))-diff.mean()
cb12=combine_freq[(((combine_count[b3]>0)&(combine_count[b2]>0))|(combine_count[b3]>=th3)|(combine_count[b2]>=th2))][[b3,b2]].replace(0,0.02)
ratb12=cb12[b3].replace(0,0.02)/cb12[b2].replace(0,0.02)
log_ratb12=np.log10(ratb12.replace(0,0.01))
p=pd.concat([log_rat12,log_ratb12],axis=1)
p.columns=['log(R3/R2) replicate 2','log(R3/R2) replicate 1']
g=sns.jointplot(x='log(R3/R2) replicate 2',y='log(R3/R2) replicate 1',color='r',data=p,alpha=0.6,size=4, annot_kws={"loc":(-0.08,0.90),'fontsize':13},joint_kws={'s':6})
g.savefig(join(savedir,'2c_consistence.pdf'),bbox_inches="tight")

print "plotting 2a"
sns.set_palette('hls', 8)
th2=5
th3=5
chart=combine_freq1[(((combine_count1[a2]>0)&(combine_count1[a3]>0))|(combine_count1[a3]>=th3)|(combine_count1[a2]>=th2))][[a2,a3]]
rat=chart[a3]/chart[a2]
tr=np.log10(rat)
pos=tr[((tr>=0)&((chart[a3]-chart[a2])>=thr))|(chart[a3]>=thr2)]
neg=tr[(tr<0)&(chart[a3]<thr2)&((chart[a2]-chart[a3])>=thr)]
train=pd.concat((pos,neg),axis=0)
train.iloc[0:len(pos)]=True
train.iloc[len(pos):]=False
class_score=all_score[all_score.columns[-5:]]
perform6=[]
mo=class_score.columns
for i in [0]:
    vali=list(set(train.index)&set(uniq2))
    label=train.loc[vali].values.astype(int)
    pred=(class_score.loc[vali]).values
    for j in range(0,len(class_score.columns)):
        auroc = roc_auc_score(label,pred[:,j])
        perform6.append([auroc,'auROC',mo[j],'single network'])
        acr=accuracy_score(label,pred[:,j]>0.5)
        perform6.append([acr,'accuracy',mo[j],'single network'])
    pred2=np.mean(pred,axis=1)
    auroc = roc_auc_score(label,pred2)
    perform6.append([auroc,'auROC','mean','ensemble_mean'])
    acr=accuracy_score(label,pred2>0.5)
    perform6.append([acr,'accuracy','mean','ensemble_mean'])
class_score['mean']=class_score[class_score.columns].mean(axis=1)
perform6pd= pd.DataFrame(perform6,columns=['Value','Metric','Model','method'])
dataset_name = 'Easy_classification_0604_freq'
perform = []
rocs = []
models = ['seq_32x1_16','seq_64x1_16','seq_32x2_16','seq_32_32','seq_32x1_16_filt3','seq_emb_32x1_16']
fig=plt.figure(figsize=(6,4))
plt.rc('font', size=20)
plt.rc('axes', titlesize=10)
plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('legend', fontsize=12)
plt.rc('figure', titlesize=20)
plt.plot([0, 1], [0, 1], 'k--',linewidth=1.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title("Classification test auROC",fontsize=15)
vali=list(set(train.index)&set(uniq2))
label=train.loc[vali].values.astype(int)
best='Easy_classification_0604_holdouttop.seq_32x2_16.score'
pred=(class_score[best].loc[vali]).values
roc= roc_curve(label,pred)
plt.plot(roc[0],roc[1],label='Lucentis single CNN(AUC = %0.3f)' % roc_auc_score(label,pred),linewidth=2)
best='mean'
pred=(class_score[best].loc[vali]).values
roc= roc_curve(label,pred)
plt.plot(roc[0],roc[1],label='Lucentis ensemble mean(AUC = %0.3f)' % roc_auc_score(label,pred),linewidth=2)
expt='Mock_b'
with open(join('../data',dataset_name,expt,'CV0', 'data.target.shuffled.test')) as f:
	label = [float(x.split()[0]) for x in f]
t_bestmodel = (None,None,-1)
for model in models:
	pred = None
	t_topdir = join('../data',dataset_name,expt,'CV0','pred.'+expt+'.'+model+'.data.test.h5.batch')
	for x in listdir(t_topdir):
	    with open(join(t_topdir,x,'0.pkl')) as f:
		pred = cPickle.load(f) if pred is None else np.append(pred,cPickle.load(f))
	b_pred = [1 if x>0.5 else 0 for x in pred ]
	t_auROC = roc_auc_score(label,pred)
	perform.append([t_auROC,'Test auROC',model,expt])
	perform.append([accuracy_score(label,b_pred),'Test accuracy',model,expt])
	rocs.append([roc_curve(label,pred),model,expt])
	if t_auROC > t_bestmodel[-1]:
	    t_bestmodel = (model,rocs[-1][0],t_auROC)
plt.plot(t_bestmodel[1][0],t_bestmodel[1][1],label='Mock CNN (AUC = %0.3f)' % t_bestmodel[2],linewidth=2.6)
plt.legend(loc="lower right")
fig.savefig(join(savedir,'2a_auroc.pdf'),bbox_inches="tight")

print "plotting 4c"
sns.set_context("paper",font_scale=1.8)
sns.set_palette('hls', 8)
pfgroup={'cnn_score':[],'log(R3/R1)':[]}
for pc in [1.0,0.0,-1.0,-1.5]:
    if pc==1.0:
        p=gifford2[(gifford2['R3/R1_score']>=pc)]
        pfgroup['log(R3/R1)']=pfgroup['log(R3/R1)']+['>=1.0']*len(p)
    elif pc==-1.5:
        p=gifford2[gifford2['R3/R1_score']<-1.0]
        pfgroup['log(R3/R1)']=pfgroup['log(R3/R1)']+['<-1.0']*len(p)
    else:
        p=gifford2[(gifford2['R3/R1_score']>=pc)&(gifford2['R3/R1_score']<(pc+1.0))]
        pfgroup['log(R3/R1)']=pfgroup['log(R3/R1)']+[str(pc+1.0)+'~'+str(pc)]*len(p)
    pfgroup['cnn_score']=pfgroup['cnn_score']+list(p['cnn_score'].values)
pfgroup=pd.DataFrame(pfgroup)
fig=plt.figure(figsize=(5,5))
ax=sns.boxplot(x='log(R3/R1)',y='cnn_score',data=pfgroup,fliersize=2)
ax.set_ylabel('Prediction (ensemble mean)')
plt.title('Enrichment (standard)')
fig.savefig(join(savedir,'4c_cnn_enrich_box_standard.pdf'),bbox_inches="tight")
pfgroup={'cnn_score':[],'log(R3/R1)':[]}
for pc in [1.0,0.0,-1.0,-1.5]:
    if pc==1.0:
        p=gifford2[(gifford2['R3/R1J_score']>=pc)]
        pfgroup['log(R3/R1)']=pfgroup['log(R3/R1)']+['>=1.0']*len(p)
    elif pc==-1.5:
        p=gifford2[gifford2['R3/R1J_score']<-1.0]
        pfgroup['log(R3/R1)']=pfgroup['log(R3/R1)']+['<-1.0']*len(p)
    else:
        p=gifford2[(gifford2['R3/R1J_score']>=pc)&(gifford2['R3/R1_score']<(pc+1.0))]
        pfgroup['log(R3/R1)']=pfgroup['log(R3/R1)']+[str(pc+1.0)+'~'+str(pc)]*len(p)
    pfgroup['cnn_score']=pfgroup['cnn_score']+list(p['cnn_score'].values)
pfgroup=pd.DataFrame(pfgroup)
fig=plt.figure(figsize=(5,5))
ax=sns.boxplot(x='log(R3/R1)',y='cnn_score',data=pfgroup,fliersize=2)
ax.set_ylabel('Prediction (ensemble mean)')
plt.title('Enrichment (stringent)')
fig.savefig(join(savedir,'4c_cnn_enrich_box_stringent.pdf'),bbox_inches="tight")

