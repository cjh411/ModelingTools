# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 18:45:27 2017

@author: christopherhedenberg
"""

import pandas as pd
import wquantiles
import numpy as np

def weighted_cuts(col,weight,nbin):
    wtd_cuts = [wquantiles.quantile(col,weight,float(x)/float(nbin)) for x in range(1,nbin) ]
    wtd_cuts_str =["(-inf,%s]"%wtd_cuts[0]]
    for i in range(len(wtd_cuts)-1):
        wtd_cuts_str.append("(%s,%s]"%(wtd_cuts[i],wtd_cuts[i+1]))
    wtd_cuts_str.append("(%s,inf]"%wtd_cuts[len(wtd_cuts)-1])
    buckets = np.take(wtd_cuts_str,np.searchsorted(wtd_cuts,col))
    return buckets
    
        
def weighted_bucket(df,col= None,weight=None,nbin=10):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    if col == None:
        try:
            to_bucket=list(df.columns.values)
        except ValueError:
            print("cannot calculate quantiles for specified columns")
    elif type(col) in [list, str]:
        if type(col) is list:
            to_bucket=col
        else:
            to_bucket=list(col)
    else:
        raise TypeError("Columns must be a list or string name")
    if type(weight)==type(None):
        weight=[1 for i in range(df.shape[0])]
    for var in to_bucket:
        if df[var].dtype in numerics:
            df["%s_grp%s"%(var,nbin)]=weighted_cuts(df[var],weight,nbin)
            df["%s_grp%s"%(var,nbin)].astype('category').cat.add_categories("Missing",inplace=True)
            df.loc[df[var].isna(),"%s_grp%s"%(var,nbin)]="Missing"
    return df
    
def impute(df,cat=True,num=True,col=None,split_by=None,num_mthd="mean",cat_mthd="dist"):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    if col == None:
        try:
            to_bucket=list(df.columns.values)
        except ValueError:
            print("cannot calculate quantiles for specified columns")
    elif type(col) in [list, str]:
        if type(col) is list:
            to_bucket=col
        else:
            to_bucket=list(col)
    else:
        raise TypeError("Columns must be a list or string name")
    to_bucket = [item for item in to_bucket if pd.isnull(df[item]).sum()>0]
    for var in to_bucket:
        if split_by==None:
            if df[var].dtype in numerics and num == True:
                df["%s_%s_imp"%(var,num_mthd)]=df[var]
                if num_mthd=="mean":
                    df["%s_%s_imp"%(var,num_mthd)].fillna(df[var].mean(),inplace=True)
                if num_mthd=="med":
                    df["%s_%s_imp"%(var,num_mthd)].fillna(df[var].median(),inplace=True)
                if num_mthd=="mode":
                    df["%s_%s_imp"%(var,num_mthd)].fillna(df[var].mode(),inplace=True)
            elif cat == True:
                df["%s_%s_imp"%(var,cat_mthd)]=df[var].astype('category')
                if cat_mthd == "dist":
                    if "Missing" not in df["%s_%s_imp"%(var,cat_mthd)].cat.categories:
                        df["%s_%s_imp"%(var,cat_mthd)].cat.add_categories("Missing",inplace=True)
                    df["%s_%s_imp"%(var,cat_mthd)].fillna("Missing",inplace=True)
                elif cat_mthd == "top":
                    agg = list(df.groupby(var).size().sort_values(ascending=False,inplace=True).index)[0]
                    df["%s_%s_imp"%(var,cat_mthd)].fillna(agg,inplace=True)
        else:
            if df[var].dtype in numerics and num == True:
                for split in list(set(df[split_by].tolist())):
                    df["%s_%s_imp"%(var,num_mthd)]=df[var]
                    if num_mthd=="mean":
                        df["%s_%s_imp"%(var,num_mthd)].fillna(df.loc[df[split_by]!=split,var].mean(),inplace=True)
                    if num_mthd=="med":
                        df["%s_%s_imp"%(var,num_mthd)].fillna(df.loc[df[split_by]!=split,var].median(),inplace=True)
                    if num_mthd=="mode":
                        df["%s_%s_imp"%(var,num_mthd)].fillna(df.loc[df[split_by]!=split,var].mode(),inplace=True)
            elif cat == True:
                df["%s_%s_imp"%(var,cat_mthd)]=df[var].astype('category')
                if cat_mthd == "dist":
                    if "Missing" not in df["%s_%s_imp"%(var,cat_mthd)].cat.categories:
                        df["%s_%s_imp"%(var,cat_mthd)].cat.add_categories("Missing",inplace=True)
                    df["%s_%s_imp"%(var,cat_mthd)].fillna("Missing",inplace=True)
                elif cat_mthd == "top":
                    agg = df.groupby(var).size()
                    agg.sort_values(ascending=False,inplace=True)
                    df["%s_%s_imp"%(var,cat_mthd)].fillna(agg.index[0],inplace=True)                  
    return df
                
                
                    
def tweedie_deviance(y,mu,p):
    dev = 2*( ( (y**(2-p))/((1-p)*(2-p))) - ((y*(mu**(1-p)))/(1-p)) - ((mu**(2-p))/(2-p)) )
    return dev  
  
def gamma_deviance(y,mu,p):
    dev = 2*( ( (y**(2-p))/((1-p)*(2-p))) - ((y*(mu**(1-p)))/(1-p)) - ((mu**(2-p))/(2-p)) )
    return dev    
    
def poisson_deviance(y,mu,p):
    dev = 2*( ( (y**(2-p))/((1-p)*(2-p))) - ((y*(mu**(1-p)))/(1-p)) - ((mu**(2-p))/(2-p)) )
    return dev    