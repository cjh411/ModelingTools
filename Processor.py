# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 18:31:20 2017

@author: christopherhedenberg
"""

import sys
sys.path.append('/Users/christopherhedenberg/Downloads/projects/DataRobotTool')

import sklearn as sk
from sklearn.datasets import load_boston
import pandas as pd
#import DataCleaner as dc
#import ModelClasses as mc
#import Utilities



housing = pd.read_csv("C:/Users/n0267335/Documents/Modeling/Tools/Python/ModelingTools/train_housing.csv")
tmp=data_cleaner(housing,split=True)
tmp.clean_data()

mdl = Model("gaussian",tmp.x.drop("SalePrice",axis=1),tmp.x["SalePrice"],mdl_by="TestInd")

glmobj = glm(mdl.family,mdl.x,mdl.y,mdl_by=mdl.mdl_by)

#glm.fit_mdl()

mdls = glmobj.fit_list(varlst=["Alley_dist_imp"])
a=mdls[10]

pred = glmobj.cv_predict_oof(glmobj.x,a)


pickle.dump(glmobj,open("/Users/christopherhedenberg/Downloads/DS/projects/DataRobotTool/TestPickle.p",'wb'))
b = pickle.load(open("/Users/christopherhedenberg/Downloads/DS/projects/DataRobotTool/TestPickle.p",'rb'))