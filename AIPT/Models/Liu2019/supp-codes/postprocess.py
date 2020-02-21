import numpy as np
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from os.path import dirname, basename,join,exists
from os import makedirs,system,listdir,popen
import seaborn as sns
sns.set(color_codes=True)
padlen=20
plt.style.use('default')
plt.style.use('seaborn-white')
sns.set_style("white")
sns.set_context("paper",font_scale=1.6)
gifford2=pd.read_csv('../data/2nd_experiment_spreadsheet_denoise.csv',index_col=0)
new_seeds_all=np.loadtxt('filtered_seed',dtype='S')
new_seeds=set([x for x in new_seeds_all if ('X' not in x)])
def check2(x):
    return x in new_seeds
gen_dir='../results/seq_gen'
c_seq=new_seeds
check=[]
temp=pd.Series([1]*len(new_seeds))
temp.index=new_seeds
check.append(temp.to_frame('C'))
for net in ['Easy_classification_0622_reg','Easy_classification_0615_holdouttop_reg','Easy_classification_0604_holdouttop_reg','Easy_classification_0604_holdouttop']:
    print net
    for model in ['seq_32x1_16', 'seq_64x1_16','seq_32x2_16','seq_32_32','seq_32x1_16_filt3','seq_emb_32x1_16']:
        for stepsize in ['0.0001','0.001','0.005','0.0005','5e-05']:#,'0.001']:
            for L in ['10','20','30','40','50','60','70']:
                pred_file = join(gen_dir,net,model,'genseq-{}-{}.tsv'.format(L,stepsize))
                pred_code = net[20:24]+net[-3:-1]+'_'+model+'_'+basename(pred_file).split('.tsv')[0]
                if not exists(pred_file):
                    continue
                if net in['Easy_classification_0604_holdouttop_reg','Easy_classification_0604_holdouttop']:
                    if model=='seq_emb_32x1_16':
                        continue
                smap=pd.read_csv(pred_file,sep='\t',index_col=0,header=None)
                smap.columns=['new','seed','own_score','sscore']
                smap=smap[smap['seed'].apply(check2)]
                raw_seq=smap['new'].values
                g_seq=np.asarray([x for x in raw_seq if ((x[-2:]=='DY')|(x[-2:]=='DV'))])
                g_seq=np.unique(g_seq)
                temp=pd.Series([1]*len(g_seq))
                temp.index=g_seq
                check.append(temp.to_frame(pred_code))
combine_check=pd.concat(check,axis=1)
combine_check=combine_check.fillna(0)
count=combine_check[combine_check['C']!=1].sum(axis=1)
final_sorted=count.sort_values(axis=0, ascending=False)
print final_sorted.loc['MHYYDIGVFPWDTFDY']
inter_dir='../results/intermediate'
if not exists(inter_dir):
    makedirs(inter_dir)
combine_check.to_csv(join(inter_dir,'genseq_vote.csv'))
top=final_sorted.index
rdir=join(inter_dir,'all_proposed_seq')
if not exists(rdir):
    makedirs(rdir)
f=open(join(rdir,'data.tsv'),'w')
flabel=open(join(rdir,'data.target'),'w')
pos_seq=list(final_sorted.index)
for i in range(len(pos_seq)):
    back='J'*((padlen-len(pos_seq[i]))/2)+pos_seq[i]+'J'*((padlen+1-len(pos_seq[i]))/2)
    f.writelines(str(i)+'\t'+back+'\n')
    flabel.writelines('1\n')
f.close()
flabel.close()
cmd=' '.join(['python','utils/embedH5.py',join(rdir,'data.tsv'),join(rdir,'data.target'),join(rdir,'data.h5'),'-m','mapper'])
cmd2=' '.join(['python','utils/compute_pred.py',join(rdir,'score'),join(rdir)])
print "converting to h5py"
system(cmd)
if exists(join(rdir,'score')):
    system('rm -r '+join(rdir,'score'))
print "predicting ensemble score"
system(cmd2)
all_score=pd.read_csv(join(rdir,'score','scores.csv'),index_col=0)
all_score=all_score.loc[final_sorted.index]
reg_score=all_score[all_score.columns[0:-5]]
new_gifford=final_sorted.to_frame('vote')
new_gifford['cnn_score']=reg_score.mean(axis=1)
new_gifford['cnn_ci']=(reg_score.std(axis=1)).values*1.96*2/np.sqrt(len(reg_score.columns))
new_gifford['5%_Gauss_lb']=new_gifford['cnn_score']-(new_gifford['cnn_ci'])/2
new_gifford['R3/R1J_score']=gifford2['R3/R1J_score'].loc[new_gifford.index]
new_gifford['R3/R1_score']=gifford2['R3/R1_score'].loc[new_gifford.index]
new_gifford['Ens_Grad2']=True
new_gifford['Ens_Grad2'][new_gifford['5%_Gauss_lb']<0]=False
new_gifford['Ens_Grad2'][new_gifford['vote']<10]=False
top=new_gifford[new_gifford['Ens_Grad2']>0].index
set1=set(gifford2[gifford2['Seed_new']==1].index)
set3=set(top)
def check3(x):
    return (x in set3)
def dist(a,b):
    return sum([a[i]!=b[i] for i in range(len(a))])
temp=pd.Series([1]*len(c_seq))
temp.index=c_seq
seedmap=pd.DataFrame({})
for net in ['Easy_classification_0622_reg','Easy_classification_0615_holdouttop_reg','Easy_classification_0604_holdouttop_reg','Easy_classification_0604_holdouttop']:
    print net
    for model in ['seq_32x1_16', 'seq_64x1_16','seq_32x2_16','seq_32_32','seq_32x1_16_filt3','seq_emb_32x1_16']:
        for stepsize in ['0.0001','0.001','0.005','0.0005','5e-05']:#,'0.001']:
            for L in ['10','20','30','40','50','60','70']:
                pred_file = join(gen_dir,net,model,'genseq-{}-{}.tsv'.format(L,stepsize))
                pred_code = net[19:23]+net[-3:-1]+'_'+model+'_'+basename(pred_file).split('.tsv')[0]
                if not exists(pred_file):
                    continue
		if net in['Easy_classification_0604_holdouttop_reg','Easy_classification_0604_holdouttop']:
                    if model=='seq_emb_32x1_16':
                        continue
                smap=pd.read_csv(pred_file,sep='\t',index_col=0,header=None)
                smap.columns=['new','seed','own_score','sscore']
                smap=smap[smap['new'].apply(check3)]
                smap=smap[smap['seed'].apply(check2)]
                smap=smap.drop_duplicates()
                smap['dataset']=net
                smap['model']=model
                smap['L']=int(L)
                smap['step']=float(stepsize)
                smap['vote']=(new_gifford['vote'].loc[smap['new']]).values
                smap['cnn_score']=(new_gifford['cnn_score'].loc[smap['new']]).values
                smap['cnn_ci']=(new_gifford['cnn_ci'].loc[smap['new']]).values
                smap['5%_Gauss_lb']=(new_gifford['cnn_ci'].loc[smap['new']]).values
                smap['R3/R1J_score']=(new_gifford['R3/R1J_score'].loc[smap['new']]).values
                smap['R3/R1_score']=(new_gifford['R3/R1_score'].loc[smap['new']]).values
                smap['R3/R1J_score_seed']=(gifford2['R3/R1J_score'].loc[smap['seed']]).values
                smap['R3/R1_score_seed']=(gifford2['R3/R1_score'].loc[smap['seed']]).values
                smap['Ens_Grad2']=(new_gifford['Ens_Grad2'].loc[smap['new']]).values
                smap['dist']=smap.apply(lambda row: dist(row['new'], row['seed']), axis=1)
                seedmap=seedmap.append(smap, ignore_index=True)
seedmap['group']="<=-1"
for left in range(-1,3,1):
    if not left==2:
        seedmap['group'][seedmap['R3/R1J_score']>left]=str(left)+'~'+str(left+1)
    else:
        seedmap['group'][seedmap['R3/R1J_score']>left]='>2'
seedmap.to_pickle(join(rdir,'data_proposed_seedmap.pkl'))
grp=seedmap.groupby('new')
compressed=pd.concat([grp['dataset'].unique(),grp['seed'].unique(),grp['R3/R1_score'].mean(),grp['R3/R1J_score'].mean(),\
                      grp['cnn_score'].mean(),grp['cnn_ci'].mean(),grp['vote'].mean(),grp['Ens_Grad2'].mean(),grp['dist'].min(),\
                      grp['5%_Gauss_lb'].mean(),grp['L'].unique(),grp['step'].unique()],axis=1)
compressed['good?']=compressed['R3/R1J_score']>0
compressed['#dataset']=compressed['dataset'].apply(len)
compressed['#step']=compressed['step'].apply(len)
compressed['#L']=compressed['L'].apply(len)
compressed['group']="<=-1"
for left in range(-1,3,1):
    if not left==2:
        compressed['group'][compressed['R3/R1J_score']>left]=str(left)+'~'+str(left+1)
    else:
        compressed['group'][compressed['R3/R1J_score']>left]='>2'
compressed.to_pickle(join(rdir,'compressed_proposed_seedmap.pkl'))
plot_dir='../results/plots'
if not exists(plot_dir):
    makedirs(plot_dir)
def plot_dist_onethre3_new(plist,palette,chart,legend,pfix):
    kd=True
    hist=False
    sns.set_palette(palette, 8)
    fig2, ax2 = plt.subplots(figsize=(6,3.5))
    fig3, ax3 = plt.subplots(figsize=(6,3.5))
    for i in range(len(plist)):
        per=plist[i]
        lgd=legend[i]
        lw=1.8
        if per in ['Ens_Grad2','Seed_new']:
                lw=1.8
        else:
                lw=1
        if per in ['Seed','Seed_new']:
            rat=chart[chart[per]>0]['R3/R1J_score']
            rat2=chart[chart[per]>0]['R3/R2J_score']
        else:
            rat=chart[(chart[per]>0)&(chart['Seed_new']==0)]['R3/R1J_score']
            rat2=chart[(chart[per]>0)&(chart['Seed_new']==0)]['R3/R2J_score']
        rat13=rat[~rat.isnull()]
        if len(rat13)>1:
            ax2=sns.distplot(rat13.values,hist=hist,kde=kd,kde_kws={"label":lgd,'lw':lw},\
                             axlabel="log(R3/R1)",ax=ax2)
        rat23=rat2[~rat2.isnull()]
        if len(rat23)>1:
            ax3=sns.distplot(rat23.values,hist=hist,kde=kd,kde_kws={"label":lgd,'lw':lw},\
                             axlabel="log(R3/R2)",ax=ax3)
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.set_title('Enrichment (stringent)',fontsize=15)
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.set_title('Enrichment (stringent)',fontsize=15)
    hand,lab=ax2.get_legend_handles_labels()
    ax2.legend(handles=(hand[0],hand[2],hand[5],hand[1],hand[3],hand[4]))
    hand,lab=ax3.get_legend_handles_labels()
    ax3.legend(handles=(hand[0],hand[2],hand[5],hand[1],hand[3],hand[4]))
    fig2.savefig(join(plot_dir,'4a_dist_shiftr1r3'+pfix+'.pdf'),bbox_inches="tight")
    fig3.savefig(join(plot_dir,'4a_dist_shiftr2r3'+pfix+'.pdf'),bbox_inches="tight")

def stack_plot(plist,palette,chart,legend,th,pfix):
    sns.set_context("paper",font_scale=1.8)
    nm=0
    kd=False
    bins1=np.arange(th,4.1,0.1)
    hist=True
    sns.set_palette(palette, 8)
    fig2, ax2 = plt.subplots(figsize=(7.8,4))
    plt.xlim(th, 4.1)
    for i in range(len(plist)):
        per=plist[i]
        lgd=legend[i]
        lw=1.8
        if per=='Seed_new':
            rat=chart[(chart[plist[i:len(plist)]]>0).sum(axis=1)>0]['R3/R1J_score']
        else:
            rat=chart[(chart[plist[i:len(plist)]]>0).sum(axis=1)>0][chart['Seed_new']==0]['R3/R1J_score']
        rat13=rat[~rat.isnull()]    
        a13=rat13[rat13>=th].values
        if len(a13)==1:
            a13=[a13,a13]
        ax2=sns.distplot(a13,hist=hist,bins=bins1,kde=kd,kde_kws={"label":lgd},hist_kws={"edgecolor":'black',"normed":nm,"label":lgd,"alpha":1},axlabel="log(R3/R1)",ax=ax2)
    ax2.legend()
    ax2.set_title('Stringent wash enrichment',fontsize=15)
    fig2.savefig(join(plot_dir,'4b_stack_R3R1'+pfix+'.pdf'),bbox_inches="tight")

gifford2['Ens_Grad2']=False
gifford2['Ens_Grad2'].loc[set(compressed.index)&set(gifford2.index)]=True
plist1=['Seed_new','GA_KNN2','Ens_Grad2','GA_CNN2','Neg_Ctrl','VAE2']
leg1=['Seed','GA_KNN','Ens_Grad','GA_CNN','Neg_Ctrl','VAE']
plot_dist_onethre3_new(plist1,"bright",gifford2,leg1,'stringent')
plist0=['Seed_new','GA_KNN2','GA_CNN2','Neg_Ctrl','VAE2','Ens_Grad2']
leg0=['Seed','GA_KNN','GA_CNN','Neg_Ctrl','VAE','Ens_Grad']
stack_plot(plist0,"hls",gifford2,leg0,1.7,'stringent')

