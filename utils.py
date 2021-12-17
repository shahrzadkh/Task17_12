import numpy as np
from typing import Tuple
import unittest as test
from sklearn.linear_model import LassoCV,LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from typing import Dict
import pickle

class DataProvider:
    def __init__(self, training_size: int, validation_size: int,test_size: int, valuerange: Tuple[int, int]):
        assert len(valuerange) == 2
        assert valuerange[1] > valuerange[0]
        self._valuerange = valuerange
        np.random.seed(0)
        self._training = self._generate_data(training_size)
        np.random.seed(10)
        self._validation = self._generate_data(validation_size)
        np.random.seed(20)
        self._test = self._generate_data(test_size)
    def _generate_data(self, length: int) -> np.ndarray:
        return self._valuerange[0] + np.random.rand(length, 6) * (self._valuerange[1] - self._valuerange[0])
        # [] gives a dict-like accessor to permit variable access;
        # names are 'training', 'validation', and 'test'
    def __getitem__(self, field_name: str) -> np.ndarray:
        if field_name == 'training':
            #np.random.seed(0)
            return self._training
        if field_name == 'validation':
            return self._validation
        if field_name == 'test':
            #np.random.seed(0)
            return self._test



class CV_Regr:


    def __init__(self, X_train: np.ndarray, Y_train: np.ndarray,
                 X_validate: np.ndarray, Y_validate: np.ndarray, n_itter: int):

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_validate = X_validate
        self.Y_validate = Y_validate
        self.n_itter = n_itter
        if not isinstance(n_itter, int):
            raise ValueError("n_itter must be int")
        if len(self.X_train.shape) > 1 and not len(self.X_train.shape) == 2:
            raise ValueError("Variable X_train must be of shape: n_sample x n_features.")
        elif len(self.X_train.shape) == 0:
            raise ValueError("missing X_train of shape: n_sample x n_features.")


        if len(self.X_validate.shape) > 1 and not len(self.X_validate.shape) == 2:
            raise ValueError("Variable X_validate must be of shape: n_sample x n_features.")
        elif len(self.X_validate.shape) == 0:
            raise ValueError("missing X_validate of shape: n_sample x n_features.")

        if len(self.Y_train.shape) > 1:
            if not len(self.Y_train.shape) == 2:
                raise ValueError("Variable Y must be an array of shape: n_sample.")
            if not self.Y_train.shape[1] == 1:
                raise ValueError("Variable Y must be an array of shape: n_sample x 1.")
        elif len(self.Y_train.shape) == 0:
            raise ValueError("missing Y of shape: n_sample x 1.")


        if len(self.Y_validate.shape) > 1:
            if not len(self.Y_validate.shape) == 2:
                raise ValueError("Variable Y_validate must be an array of shape: n_sample.")
            if not self.Y_validate.shape[1] == 1:
                raise ValueError("Variable Y_validate must be an array of shape: n_sample x 1.")
        elif len(self.Y_validate.shape) == 0:
            raise ValueError("missing Y_validate of shape: n_sample x 1.")

        if not self.X_train.shape[0] == self.Y_train.shape[0]:
            raise ValueError("The first dimension of X and Y should be the same.")
        if not self.X_validate.shape[0] == self.Y_validate.shape[0]:
            raise ValueError("The first dimension of X_validate and Y_validate should be the same.")


        Std_obj = StandardScaler()
        self.pipeline_steps =[("standardizing",Std_obj)]
        self._Gridsearch_dict = {'GS_option':False, 'param_grid':dict()}

    def _nested_CV(self, random_state: int):
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=random_state) # hyperparameter
        outer_cv = KFold(n_splits=4, shuffle=True, random_state=random_state) # This would provide validated outcomes
        return inner_cv, outer_cv

    def train_Regressor(self, estimator = 'linear_reg'):
        if estimator is 'linear_reg':
            reg_obj, training_score = self._linear_reg()
            self.reg_obj=reg_obj
        elif estimator is 'randomforest_reg':
            reg_obj, training_score = self._randomforest_reg()
            self.reg_obj=reg_obj
        elif estimator is 'lassocv_reg':
            reg_obj, training_score = self._lassocv_reg()
            self.reg_obj=reg_obj

        elif estimator is 'lasso_reg':
            reg_obj, training_score = self._lasso_reg()
            self.reg_obj=reg_obj
        return reg_obj, training_score

    def validate_Regressor(self, X_test: np.ndarray, Y_test: np.ndarray):
        yhat = self.reg_obj.predict(X_test)
        score = {'r2':[], 'mae':[]}
        score['r2'] = r2_score(Y_test, yhat)
        score['mae'] = mean_absolute_error(Y_test, yhat)
        return score

    def predict_Regressor(self, X_test: np.ndarray):
        yhat = self.reg_obj.predict(X_test)
        return yhat



    def _gsCV(self, estimator):
        base_estimator_score=[]
        outer_results = {'r2':[], 'mae':[]}

        inner_cv, _ = self._nested_CV(42)
        if self._Gridsearch_dict['GS_option'] == True:
            estimator_obj = GridSearchCV(estimator, self._Gridsearch_dict['param_grid'], n_jobs=-1, cv= inner_cv)

        else:
            estimator_obj = estimator
        r2=[]
        mae= []

        result = estimator_obj.fit(self.X_train, self.Y_train)
        if self._Gridsearch_dict['GS_option'] == True:
                    # get the best performing model fit on the whole training set
            best_model = result.best_estimator_
            print(result.best_params_)
        else:
            best_model = result
        # evaluate model on the hold out dataset
        self.reg_obj=best_model
        validation_score = self.validate_Regressor(self.X_validate, self.Y_validate)

        return best_model , validation_score


    def _linear_reg(self):
        lr_obj = LinearRegression()
        pipeline_steps= self.pipeline_steps.copy()
        pipeline_steps.append(("Linear_Reg", lr_obj))
        pipe= Pipeline(pipeline_steps)
        self._Gridsearch_dict = {'GS_option':False, 'param_grid':dict()}
        model , outer_results = self._gsCV(pipe)
        score = {'r2':[], 'mae':[]}
        score['r2']= np.mean(outer_results['r2'])
        score['mae']= np.mean(outer_results['mae'])
        Out_Obj = model.fit(self.X_train, self.Y_train)

        print(Out_Obj)
        return Out_Obj, score


    def _randomforest_reg(self):
        rf_obj = RandomForestRegressor(random_state=0)
        pipeline_steps= self.pipeline_steps.copy()
        pipeline_steps.append(("RF_obj", rf_obj))
        self._Gridsearch_dict = {'GS_option':True,'param_grid':{"RF_obj__n_estimators": [10, 20]}}##,50, 100, 200, 300

        pipe= Pipeline(pipeline_steps)
        model , outer_results = self._gsCV(pipe)
        score = {'r2':[], 'mae':[]}
        score['r2']= np.mean(outer_results['r2'])
        score['mae']= np.mean(outer_results['mae'])
        Out_Obj = model.fit(self.X_train, self.Y_train)
        print(Out_Obj)
        return Out_Obj, score

    def _lassocv_reg(self):
        lassoCV_obj = LassoCV(cv = 3)
        pipeline_steps= self.pipeline_steps.copy()
        pipeline_steps.append(("LassoCV_obj", lassoCV_obj))
        self._Gridsearch_dict = {'GS_option':False, 'param_grid':dict()}
        pipe= Pipeline(pipeline_steps)
        model , outer_results = self._gsCV(pipe)
        score = {'r2':[], 'mae':[]}
        score['r2']= np.mean(outer_results['r2'])
        score['mae']= np.mean(outer_results['mae'])
        Out_Obj = model.fit(self.X_train, self.Y_train)

        print(Out_Obj)
        return Out_Obj, score

    def _lasso_reg(self):
        lasso_obj = Lasso()
        pipeline_steps= self.pipeline_steps.copy()
        pipeline_steps.append(("Lasso_obj", lasso_obj))
        self._Gridsearch_dict= {'GS_option':True,
                                'param_grid':{"Lasso_obj__alpha": np.logspace(-4, -0.5, 30)}}

        pipe= Pipeline(pipeline_steps)
        model , outer_results = self._gsCV(pipe)
        score = {'r2':[], 'mae':[]}
        score['r2']= np.mean(outer_results['r2'])
        score['mae']= np.mean(outer_results['mae'])
        Out_Obj = model.fit(self.X_train, self.Y_train)

        print(Out_Obj)
        return Out_Obj, score

    def _generate_nested_clf_scores(self, estimator):
        base_estimator_score=[]
        outer_results = {'r2':[], 'mae':[]}
        for i in range(self.n_itter):
            inner_cv, outer_cv = self._nested_CV(i)
            if self._Gridsearch_dict['GS_option'] == True:
                estimator_obj = GridSearchCV(estimator, self._Gridsearch_dict['param_grid'], n_jobs=-1, cv= inner_cv)

            else:
                estimator_obj = estimator
            r2=[]
            mae= []
            for train_ix, Validation_ix in outer_cv.split(self.X_train):
                # split data
                X_t, X_v = self.X_train[train_ix, :], self.X_train[Validation_ix, :]
                y_t, y_v = self.Y_train[train_ix], self.Y_train[Validation_ix]
                # fit
                result = estimator_obj.fit(X_t, y_t)
                if self._Gridsearch_dict['GS_option'] == True:
                    # get the best performing model fit on the whole training set
                    model = result.best_estimator_
                else:
                    best_model = result
                # evaluate model on the hold out dataset
                yhat = model.predict(X_v)
                # evaluate the model
                r2.append(r2_score(y_v, yhat))
                mae.append(mean_absolute_error(y_v, yhat))
            outer_results['r2'].append(np.mean(r2))
            outer_results['mae'].append(np.mean(mae))


        return best_model , outer_results
