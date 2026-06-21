
import numpy as np
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class XGBoost(BaseEstimator, RegressorMixin):
  
    def __init__(self, random_state=42, n_estimators=100, max_depth=3, learning_rate=0.1, **kwargs):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.kwargs = kwargs
      
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.model = None
        self.is_multioutput = False

    def get_params(self, deep=True):
   
        params = {
            "random_state": self.random_state,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
        }
        params.update(self.kwargs)
        return params

    def set_params(self, **params):

        for key, value in params.items():
            setattr(self, key, value)
            if key not in ["random_state", "n_estimators", "max_depth", "learning_rate"]:
                self.kwargs[key] = value
        return self

    def fit(self, X, y):
      
        return self.train(X, y)

    def train(self, params, ys, y0s=None):
    
  
        params = np.array(params)
        ys = np.array(ys)
        
     
        X = params
        if y0s is not None:
            y0s = np.array(y0s)
         
            if y0s.ndim == 1:
                y0s = y0s.reshape(-1, 1)
            X = np.concatenate([X, y0s], axis=1)
    
        if ys.ndim == 3:
           
            y_target = ys[:, -1, :]
        elif ys.ndim == 2:
           
            y_target = ys
        else:
            raise ValueError(f"ys should be 2D or 3D array, got shape {ys.shape}")

       
        
        n_outputs = y_target.shape[1] if y_target.ndim > 1 else 1
        
        base_model = xgb.XGBRegressor(
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            **self.kwargs
        )

        if n_outputs > 1:
            self.is_multioutput = True
            self.model = MultiOutputRegressor(base_model)
        else:
            self.is_multioutput = False
            self.model = base_model
          
            if y_target.ndim == 2 and y_target.shape[1] == 1:
                y_target = y_target.ravel()

        self.model.fit(X, y_target)
        print(f"XGBoost training finished. Input features: {X.shape[1]}, Output targets: {n_outputs}")
        return self

    def predict(self, params, y0s=None):
   
        params = np.array(params)
        
        X = params
        if y0s is not None:
            y0s = np.array(y0s)
            if y0s.ndim == 1:
                y0s = y0s.reshape(-1, 1)
            X = np.concatenate([X, y0s], axis=1)
            
        pred = self.model.predict(X)
        
     
        if params.ndim == 1:
          
             if isinstance(pred, np.ndarray) and pred.ndim == 1 and not self.is_multioutput:
                 pred = np.array([pred]) 
             elif pred.ndim == 1 and self.is_multioutput:
                  pred = pred.reshape(1, -1)

        if not self.is_multioutput and pred.ndim == 1:
            pred = pred.reshape(-1, 1)
            
        return pred

    def __call__(self, params, y0s=None):
        return self.predict(params, y0s)
