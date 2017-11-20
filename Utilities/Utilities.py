# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 18:45:27 2017

@author: christopherhedenberg
"""

import pandas as pd
import wquantiles

class GsDF(pd.DataFrame):
    def weighted_cuts(self,col,weight,nbin):
        wtd_cuts = [wquantiles.quantile(col,weight,x/nbin) for x in range(nbin) ]
        wtd_cuts_str =["(-inf,%s]"%wtd_cuts[0]]
        inds = np.searchsorted(wtd_cuts,self[var])
        for i in range(1,len(wtd_cuts)-1):
            wtd_cuts_str.append("(%s,%s]"%(wtd_cuts[i-1],wtd_cuts[i]))
        
            
    def weighted_bucket(self,col= None,weight=None,nbin=10):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        if col == None:
            try:
                to_bucket=list(self.columns.values)
            except ValueError:
                print("cannot calculate quantiles for specified columns")
        elif type(col) in [list, str]:
            if type(col) is list:
                to_bucket=col
            else:
                to_bucket=list(col)
        else:
            raise TypeError("Columns must be a list or string name")
        for var in to_bucket:
            if self[var].dtype in numerics:
                self["%s_grp%s"%(var,nbin)]=pd.cut(self[var],nbin)
                self["%s_grp%s"%(var,nbin)].cat.add_categories("Missing",inplace=True)
                self["%s_grp%s"%(var,nbin)].fillna("Missing",inplace=True)
        return self
        
    def impute(self,cat=True,num=True,col=None,split_by=None,num_mthd="mean",cat_mthd="dist"):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        if col == None:
            try:
                to_bucket=list(self.columns.values)
            except ValueError:
                print("cannot calculate quantiles for specified columns")
        elif type(col) in [list, str]:
            if type(col) is list:
                to_bucket=col
            else:
                to_bucket=list(col)
        else:
            raise TypeError("Columns must be a list or string name")
        to_bucket = [item for item in to_bucket if pd.isnull(self[item]).sum()>0]
        for var in to_bucket:
            if split_by==None:
                if self[var].dtype in numerics and num == True:
                    self["%s_%s_imp"%(var,num_mthd)]=self[var]
                    if num_mthd=="mean":
                        self["%s_%s_imp"%(var,num_mthd)].fillna(self[var].mean(),inplace=True)
                    if num_mthd=="med":
                        self["%s_%s_imp"%(var,num_mthd)].fillna(self[var].median(),inplace=True)
                    if num_mthd=="mode":
                        self["%s_%s_imp"%(var,num_mthd)].fillna(self[var].mode(),inplace=True)
                elif cat == True:
                    self["%s_%s_imp"%(var,cat_mthd)]=self[var].astype('category')
                    if cat_mthd == "dist":
                        if "Missing" not in self["%s_%s_imp"%(var,cat_mthd)].cat.categories:
                            self["%s_%s_imp"%(var,cat_mthd)].cat.add_categories("Missing",inplace=True)
                        self["%s_%s_imp"%(var,cat_mthd)].fillna("Missing",inplace=True)
                    elif cat_mthd == "top":
                        agg = list(self.groupby(var).size().sort_values(ascending=False,inplace=True).index)[0]
                        self["%s_%s_imp"%(var,cat_mthd)].fillna(agg,inplace=True)
            else:
                if self[var].dtype in numerics and num == True:
                    for split in list(set(self[split_by].tolist())):
                        self["%s_%s_imp"%(var,num_mthd)]=self[var]
                        if num_mthd=="mean":
                            self["%s_%s_imp"%(var,num_mthd)].fillna(self.loc[self[split_by]!=split,var].mean(),inplace=True)
                        if num_mthd=="med":
                            self["%s_%s_imp"%(var,num_mthd)].fillna(self.loc[self[split_by]!=split,var].median(),inplace=True)
                        if num_mthd=="mode":
                            self["%s_%s_imp"%(var,num_mthd)].fillna(self.loc[self[split_by]!=split,var].mode(),inplace=True)
                elif cat == True:
                    self["%s_%s_imp"%(var,cat_mthd)]=self[var].astype('category')
                    if cat_mthd == "dist":
                        if "Missing" not in self["%s_%s_imp"%(var,cat_mthd)].cat.categories:
                            self["%s_%s_imp"%(var,cat_mthd)].cat.add_categories("Missing",inplace=True)
                        self["%s_%s_imp"%(var,cat_mthd)].fillna("Missing",inplace=True)
                    elif cat_mthd == "top":
                        agg = self.groupby(var).size()
                        agg.sort_values(ascending=False,inplace=True)
                        self["%s_%s_imp"%(var,cat_mthd)].fillna(agg.index[0],inplace=True)                  
        return self
                    
                    
                    
def tweedie_deviance(y,mu,p):
    dev = 2*( ( (y**(2-p))/((1-p)*(2-p))) - ((y*(mu**(1-p)))/(1-p)) - ((mu**(2-p))/(2-p)) )
    return dev  
  
def gamma_deviance(y,mu,p):
    dev = 2*( ( (y**(2-p))/((1-p)*(2-p))) - ((y*(mu**(1-p)))/(1-p)) - ((mu**(2-p))/(2-p)) )
    return dev    
    
def poisson_deviance(y,mu,p):
    dev = 2*( ( (y**(2-p))/((1-p)*(2-p))) - ((y*(mu**(1-p)))/(1-p)) - ((mu**(2-p))/(2-p)) )
    return dev    