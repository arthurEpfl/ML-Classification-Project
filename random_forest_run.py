from implementations import *
from preprocess import *
import yaml

# loading configuration
with open('./config_random_forest.yml', 'r') as f:
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


def evaluate_f1_score(y, tx, forest):
    predicted_labels = forest.predict_label(tx)
    return F1_score(y, predicted_labels)


def evaluate_accuracy(y, tx, forest):
    predicted_labels = forest.predict_label(tx)
    return accuracy(y, predicted_labels)


# forest, (acc, f1) = random_forest(y_train, x_train, 3, 'entropy', 100)

loss_acc, loss_f1, forest = cross_validation(y_train,
                                             x_train,
                                             config['number_of_folds'],
                                             additional_0_y,
                                             additional_0_X,
                                             config['number_of_folds'],
                                             random_forest,
                                             [config['depth'],
                                              config['type_of_loss'],
                                              config['number_of_trees'],
                                              config['1_prediction_threshold']],
                                             evaluate_accuracy,
                                             evaluate_f1_score)

with open('history_random_forest.txt', "a") as f:
    f.write(str(config) + '\n')
    f.write("averaged over all folds accuracy and f1 score: " + str(loss_acc) + ', ' + str(loss_f1) + '\n\n')
    f.close()

x_test = preprocess_test(x_test_load,
                         config['threshold_nan'],
                         config['threshold_correlation'],
                         indices_of_used_columns)

res_y = forest.predict_label(x_test)
res_y[np.where(res_y == 0)] = -1

create_csv_submission(test_ids, res_y, "best_submission_forests")
