from implementations import *
from preprocess import *
import yaml

# loading configuration
with open('./config_reg_logistic_regression.yml', 'r') as f:
    config = yaml.safe_load(f)
    f.close()

print(config)

x_train_load, x_test_load, y_train_load, train_ids, test_ids = load_csv_data('./data/dataset_to_release/')
y_train_load[np.where(y_train_load == -1)] = 0

x_train, indices_of_used_columns = preprocess_train(x_train_load,
                                                    y_train_load,
                                                    config['threshold_nan'],
                                                    config['threshold_corr_to_y'])


# resampling X, Y with additional zero labeled data for stratification
(x_train, y_train), (additional_0_X, additional_0_y) = fit_resample(x_train,
                                                                    y_train_load,
                                                                    config['resampling_threshold'])

# new class distribution
print('y_train shape:', y_train.shape)
print('x_train shape:', x_train.shape)
print('number of ones and zeros in the resampling dataset: ',
      len(np.where(y_train == 1)[0]), len(np.where(y_train == 0)[0]))

x_train, _, _ = standardize(x_train)
additional_0_X, _, _ = standardize(additional_0_X)
x_train = np.c_[np.ones((x_train.shape[0], 1)), x_train]
additional_0_X = np.c_[np.ones((additional_0_X.shape[0], 1)), additional_0_X]


def evaluate_f1_score(y, tx, w_star):
    predicted_prob = sigmoid(tx.dot(w_star.T))
    predicted_labels = np.zeros(y.shape[0])
    predicted_labels[np.where(predicted_prob >= config['1_prediction_threshold'])] = 1
    return F1_score(y, predicted_labels)


def evaluate_accuracy(y, tx, w_star):
    predicted_prob = sigmoid(tx.dot(w_star.T))
    predicted_labels = np.zeros(y.shape[0])
    predicted_labels[np.where(predicted_prob >= config['1_prediction_threshold'])] = 1
    return accuracy(y, predicted_labels)


loss_acc, loss_f1, w = cross_validation(y_train,
                                        x_train,
                                        config['resampling_threshold'],
                                        additional_0_y,
                                        additional_0_X,
                                        config['number_of_folds'],
                                        reg_logistic_regression,
                                        [config['lambda_'],
                                         np.array([1 for i in range(x_train.shape[1])]).astype('float64'),
                                         config['max_iters'],
                                         config['learning_rate']],
                                        evaluate_accuracy,
                                        evaluate_f1_score)

with open('history.txt', "a") as f:
    f.write(str(config) + '\n')
    f.write("averaged over all folds accuracy and f1 score: " + str(loss_acc) + ', ' + str(loss_f1) + '\n\n')
    f.close()

x_test = preprocess_test(x_test_load,
                         config['threshold_nan'],
                         indices_of_used_columns)

x_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]

y_pred = sigmoid(x_test.dot(w.T))
res_y = -np.ones(y_pred.shape[0])
res_y[np.where(y_pred >= 0.6)] = 1

create_csv_submission(test_ids, res_y, "best_submission_rlr")
