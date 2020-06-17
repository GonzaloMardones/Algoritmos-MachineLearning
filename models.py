import pandas as pd 
import numpy as np

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from utils import Utils

class Models:
        #constructor
        def __init__(self):
            self.regressor = {
                'SVR': SVR(),
                'GRADIENT': GradientBoostingRegressor()
            }

            self.params = {
                'SVR':{
                    'kernel': ['linear','poly','rbf'],
                    'gamma':['auto','scale'],
                    'C':[1,5,10]
                },
                'GRADIENT':{
                    'loss':['ls','lad'],
                    'learning_rate':[0.01,0.05,0.1]
                }
            }
        def grid_training(self,X,y):
            best_score = 999
            best_model = None
            
            for name, regressor in self.regressor.items():
                
                grid_reg = GridSearchCV(regressor, self.params[name], cv=3).fit(X,y.values.ravel())
                score = np.abs(grid_reg.best_score_)

                if score < best_score:
                    best_score = score
                    best_model = grid_reg.best_estimator_
            
            utils = Utils()
            utils.model_export(best_model, best_score)

