# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 21:31:24 2017

@author: christopherhedenberg
"""

class model_manager():
    def __init__(self,x=None,y=None,resp=None,cv=None,split_by=None,split=False,save_model=False,model_path=None,split_parent = None):
        self.split=split
        self.x=x
        self.split_by=split_by
        self.split_parent=split_parent
        self.y=y
        self.kfold=None
        self.folds=None
        self.save_model=model_path
        self.model_path=model_path
        if self.split==True:
            self.cv="TestInd"
        else:
            self.cv=cv
        

                

    
            

            

            