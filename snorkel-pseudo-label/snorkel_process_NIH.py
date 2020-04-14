import os 
import pandas as pd
from tqdm import tqdm
from snorkel.labeling import PandasLFApplier,LabelModel,LFAnalysis,LabelingFunction
import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def build_raw_data (filepath):
    allfile=pd.read_csv(filepath)
    #get alldoucment
    allfile['abstract']=allfile.abstract.astype(str)
    #get allsentence
    #allabstract=[]
    allsent=[]
    allid=[]
    for i in tqdm(range(len(allfile))):
        temp=allfile.abstract.iloc[i]
        temp=sent_tokenize(temp)
        for j in range(len(temp)):
            allsent.append(temp[j])
            allid.append(allfile.pid.iloc[i])
            #allabstract.append(allfile.abstract.iloc[i])
    allsent=pd.DataFrame(allsent,columns=['sent'])
    allsent['pid']=allid
    #allsent['abstract']=allabstract
    return allfile, allsent

def loop_labing(keylist,valuelist,virus):
    def keyword_lookup(x, keywords, virus ,label):
        if any(word in x.sent for word in keywords) and any(word in x.sent for word in virus) :
            return label
        return Norelevent

    def make_keyword_lf(keywords, virus,name,label=None):
        return LabelingFunction(
            name=f"keyword_{name}",
            f=keyword_lookup,
            resources=dict(keywords=keywords,virus=virus ,label=label),
        )
    
    def abstract_lookup(x, keywords,label):
        if any(word in x.sent for word in keywords):
            return label
        return Norelevent

    def make_abstract_lf(keywords,name,label=None):
        return LabelingFunction(
            name=f"abstract_{name}",
            f=abstract_lookup,
            resources=dict(keywords=keywords,label=label),
        )
    
    Norelevent = -1
    allweaklabf=[]
    viruselist=virus
    for i in range(len(keylist)):
        vbname=keylist[i]
        vbname1=keylist[i]+'ab'
        vbnameab=vbname+'su'
        vbnameab1=vbname+'su1'
        labelvalue=i
        vblabelname=vbname+'l'
     
        globals()[vbname] = make_keyword_lf(keywords=valuelist[i],virus=viruselist,name=vbnameab,label=labelvalue)
        globals()[vbname1] = make_abstract_lf(keywords=valuelist[i],name=vbnameab1,label=labelvalue)
        allweaklabf.append(globals()[vbname])
        allweaklabf.append(globals()[vbname1])
    
    return allweaklabf

def snorkel_process (keylist,dataframe,allweaklabf):
    def func(x):
        idx = (-x).argsort()[1:]
        x[idx] = 0
        return x
    cardinalitynu=len(keylist)
    applier = PandasLFApplier(lfs=allweaklabf)
    all_train_l = applier.apply(df=dataframe)
    report=LFAnalysis(L=all_train_l, lfs=allweaklabf).lf_summary()
    print(report)
    label_model = LabelModel(cardinality=cardinalitynu,verbose=False)
    label_model.fit(all_train_l)
    predt=label_model.predict(all_train_l)
    predt1=label_model.predict_proba(all_train_l)
    keylist1=keylist.copy()
    #keylist1.append('Not_relevent')
    predt2=pd.DataFrame(predt1,columns=keylist1 )
    dataframe['L_label']=predt
    dataframe1=dataframe.join(predt2, how='outer')
    dataframe1=dataframe1[dataframe1.L_label>=0]
  
    train,test=train_test_split(dataframe1,test_size=0.2)
    
    trainsent=train.sent.values
    trainlabel=train[keylist].values
    trainlabe2=trainlabel.copy()
    np.apply_along_axis(func,  1, trainlabe2)
    trainlabe2=np.where(trainlabe2 > 0, 1, 0)
    testsent=test.sent.values
    testlabel=test[keylist].values
    testlabe2=testlabel.copy()
    np.apply_along_axis(func,  1, testlabe2)
    testlabe2=np.where(testlabe2 > 0, 1, 0)
    return trainsent,trainlabe2,testsent,testlabe2,keylist,report




