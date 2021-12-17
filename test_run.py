import os
__location__ = os.getcwd()
print(__location__)
import sys
sys.path.insert(1, '/Users/skharabian/sciebo/CV_2020/Interviews/Zühlke/sharing/')
from utils import DataProvider,CV_Regr
import pickle
import json


def run_example():
    data= DataProvider(training_size=5000, validation_size=2000, test_size=5000,valuerange=(5, 8))


    # Where to save the data:
    saving_dir_fullpath = '/Users/skharabian/sciebo/CV_2020/Interviews/Zühlke/sharing/results'
    data_file= os.path.join(saving_dir_fullpath, 'data.pkl')
    with open(data_file, 'wb') as fp:
        pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)



    log_dict = {'data_fullpath':'', 'estimator_info':{}}



    # choose the training data:
    X_train = data["training"][:, :-1]
    Y_train = data["training"][:, -1::]
    X_test = data["test"][:, :-1]
    Y_test = data["test"][:, -1::]
    X_validate = data["validation"][:, :-1]
    Y_validate = data["validation"][:, -1::]

    # save stuff in the log-file
    log_dict['data_fullpath'] = data_file




    # initialise the CV_reg object
    CV_reg_object = CV_Regr(X_train, Y_train, X_validate, Y_validate, n_itter=2)
    # train with chosen estimator
    for estimator in ['linear_reg','randomforest_reg', 'lasso_reg', 'lassocv_reg']:
        model, training_score = CV_reg_object.train_Regressor(estimator = estimator)
        test_score = CV_reg_object.validate_Regressor(X_validate, Y_validate)
        print(f"{estimator}: training_score: {training_score}, test_score:{test_score}")
        model_file = os.path.join(saving_dir_fullpath, estimator + '.pkl')
        pickle.dump(model, open(model_file, 'wb'))
        # save stuff in the log-file
        log_dict['estimator_info'][estimator]={}
        log_dict['estimator_info'][estimator]['model_fullpath'] = model_file
        log_dict['estimator_info'][estimator]['model_parameters'] = f"{model}"
        log_dict['estimator_info'][estimator]['training_scores'] = training_score
        log_dict['estimator_info'][estimator]['test_scores'] = test_score
        y_hat = CV_reg_object.predict_Regressor(X_validate)




    # Now save log-file to disk:
    with open(os.path.join(saving_dir_fullpath, 'MODEL_INFO.json'), 'w') as fp:
      json.dump(log_dict, fp, sort_keys=True, indent=4)



    # some time later...

    # load the model from disk
    #loaded_model = pickle.load(open(filename, 'rb'))

    return os.path.join(saving_dir_fullpath, 'MODEL_INFO.json')


if __name__ == '__main__':
    log_file = run_example()
    print(log_file)
