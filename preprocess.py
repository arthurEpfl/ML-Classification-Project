import numpy as np
from helpers import *


# Remove columns with standard_deviation = 0, first step because otherwise impossible to apply standardise function
def remove_std_dev_0(x):
    """
    This function removes features with a standard devaition equal to 0

    Args:
        x : matrix of dataset
    Returns:
        x_std_notzero : x matrix without columns where std is 0
        zero_std_indices : indices where std is 0

    """
    std_dev = np.std(x, axis=0)
    non_zero_std_indices = np.where(std_dev != 0)[0]
    zero_std_indices = np.where(std_dev == 0)[0]
    x_std_notzero = x[:, non_zero_std_indices]

    return x_std_notzero, zero_std_indices


# Remove columns deemed "useless", mainly administrative type data or features with only 1 or nan values possible
def remove_useless(x):
    """
    This function removes features deemed useless by observation of the codebook

    Args:
        x : matrix of dataset
    Returns:
        x_keep : x matrix without columns deemed useless
        indices_removed : indices of features that where removed

    """
    indices_removed = []

    def append_indices(n):
        indices_removed.append(n)

    append_indices(1)  # count starts at 2 (0 is 2 in excel) (-2)
    append_indices(2)
    append_indices(3)
    append_indices(4)
    append_indices(5)
    # keep DISPCODE (interview completed or not)
    append_indices(7)  # SEQNO
    append_indices(8)
    append_indices(9)
    append_indices(10)
    append_indices(11)  # COLGHOUS
    append_indices(13)
    append_indices(18)  # CTELNUM1
    append_indices(19)
    append_indices(21)  # PVTRESD2
    append_indices(22)
    append_indices(25)  # HHADULT
    append_indices(88)  # EXRACT11
    append_indices(91)  # EXRACT21
    append_indices(216)  # QSTVER
    append_indices(217)  # QSTLANG (maybe useful but highly doubt it)
    # _FRT16 ???

    x_keep = np.delete(x, indices_removed, axis=1)
    return x_keep, indices_removed


# New remove useless
def remove_useless_new(x):
    indices_to_remove = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 216, 217, 219, 220, 221, 222, 226,
                         227, 228, 229, 246, 247, 250]
    for i in indices_to_remove:
        x = np.delete(x, i, axis=1)

    # change date time exerhmm1 and exerhmm2
    def change_date_format(feature):
        for i in range(len(feature)):
            hours = (feature[i] // 100)
            minutes = feature[i] - hours * 100
            feature[i] = 60 * hours + minutes
        return feature

    x[:, 90] = change_date_format(x[:, 90])
    x[:, 91] = change_date_format(x[:, 91])

    # for weight 2 and height 3
    def change_metrics_units(feature):
        for i in range(len(feature)):
            if (feature[i] // 1000) == 9:
                x_kg = feature[i] - (feature[i] // 1000) * 1000
                feature[i] = x_kg * 2.20462262
        return feature

    x[:, 62] = change_metrics_units(x[:, 62])
    x[:, 63] = change_metrics_units(x[:, 63])

    # sec 10 columns
    def change_to_per_day(feature):
        for i in range(len(feature)):
            if feature[i] == 555:
                feature[i] = 0
            elif (feature[i] // 100) == 1:
                feature[i] = feature[i] - (feature[i] // 100) * 100
            elif (feature[i] // 100) == 2:
                value = feature[i] - (feature[i] // 100) * 100
                feature[i] = value / 7
            elif (feature[i] // 100) == 3:
                value = feature[i] - (feature[i] // 100) * 100
                feature[i] = value / (7 * 30)
            elif feature[i] == 777 or feature[i] == 999:
                feature[i] = np.nan
        return feature

    x[:, 81] = change_to_per_day(x[:, 81])
    x[:, 82] = change_to_per_day(x[:, 82])
    x[:, 83] = change_to_per_day(x[:, 83])
    x[:, 84] = change_to_per_day(x[:, 84])
    x[:, 85] = change_to_per_day(x[:, 85])
    x[:, 86] = change_to_per_day(x[:, 86])

    # alcday3, ...
    def change_to_per_day2(feature):
        for i in range(len(feature)):
            if feature[i] == 888:
                feature[i] = 0
            elif (feature[i] // 100) == 1:
                value = feature[i] - (feature[i] // 100) * 100
                feature[i] = value / 7
            elif (feature[i] // 100) == 2:
                value = feature[i] - (feature[i] // 100) * 100
                feature[i] = value / (7 * 30)
            elif feature[i] == 777:
                feature[i] = np.nan
            elif feature[i] == 999:
                feature[i] = np.nan
        return feature

    x[:, 77] = change_to_per_day2(x[:, 77])
    x[:, 94] = change_to_per_day2(x[:, 94])
    x[:, 92] = change_to_per_day2(x[:, 92])
    x[:, 89] = change_to_per_day2(x[:, 89])

    def replace_8by10(feature):
        feature[feature == 8] = 10
        return feature

    x[:, 60] = replace_8by10(x[:, 60])
    x[:, 58] = replace_8by10(x[:, 58])

    def replace_agebyage(feature):
        feature[feature == 88] = 89
        feature[feature == 77] = 78
        feature[feature == 99] = 98
        return feature

    x[:, 248] = replace_agebyage(x[:, 248])

    return x, indices_to_remove


# Replace 888,88,8 values with 0, in many features this corresponds to "None" type answer
def eight_answers_to_zero(tx):
    """
    This function replaces 888,88,8 values by 0, these numbers correspond to "None" type answer for many features

    Args:
        tx : matrix of dataset
    Returns:
        new_tx : x matrix with 0's instead of 888,88 or 8 values

    """
    new_tx = tx.copy()
    for i in range(0, tx.shape[1]):
        if 888 in new_tx[:, i]:
            new_tx[:, i][new_tx[:, i] == 888] = 0
        elif 88 in new_tx[:, i]:
            new_tx[:, i][new_tx[:, i] == 88] = 0
        elif 8 in new_tx[:, i]:
            new_tx[:, i][new_tx[:, i] == 8] = 0
    return new_tx


# Replace 999,99,9 values with nan, in mnay features this corresponds to "Refused" type answer
def nine_answers_to_nan(tx):
    """
    This function replaces 999,99,9 values by nan, these numbers correspond to "Refused" type answer for many features

    Args:
        tx : matrix of dataset
    Returns:
        new_tx : x matrix with nan's instead of 999,99 or 9 values

    """
    new_tx = tx.copy()
    for i in range(tx.shape[1]):
        if 999 in new_tx[:, i]:
            new_tx[:, i][new_tx[:, i] == 999] = np.nan
        elif 99 in new_tx[:, i]:
            new_tx[:, i][new_tx[:, i] == 99] = np.nan
        elif 9 in new_tx[:, i]:
            new_tx[:, i][new_tx[:, i] == 9] = np.nan
    return new_tx


# Replace 777,77,7 values with nan, in many features this corresponds to "Don't know/Not sure" type answer
def seven_answers_to_nan(tx):
    """
    This function replaces 777,77,7 values by nan, these numbers correspond to "Don't know/Not sure" type answer for many features

    Args:
        tx : matrix of dataset
    Returns:
        new_tx : x matrix with nan's instead of 777,77 or 7 values

    """
    new_tx = tx.copy()
    for i in range(tx.shape[1]):
        if 777 in new_tx[:, i]:
            new_tx[:, i][new_tx[:, i] == 777] = np.nan
        elif 77 in new_tx[:, i]:
            new_tx[:, i][new_tx[:, i] == 77] = np.nan
        elif 7 in new_tx[:, i]:
            new_tx[:, i][new_tx[:, i] == 7] = np.nan
    return new_tx


def standardize_nan_preserved1(x):
    """
    This function standardizes a dataset with nan values by keeping the nan value as nan

    Args:
        x : matrix of dataset
    Returns:
        standardized_data : x standardized matrix with nan values kept as nan

    """
    nan_mask = np.isnan(x)
    means = np.nanmean(x, axis=0)
    centered_data = x - means
    std_dev = np.nanstd(x, axis=0)

    with np.errstate(invalid='ignore'):  # Ignore warnings due to invalid value encountered in divide
        standardized_data = centered_data / std_dev

    standardized_data[nan_mask] = np.nan  # Restore NaN values

    return standardized_data


# Remove features of dataset that exceed certain threshold of nan values
# Need to figure out why the ouptut shape different if input is standardised or not
def remove_high_nan_features(tx, threshold):
    """
    This function removes features with a certain threshold percentage of nan values

    Args:
        tx : matrix of dataset
        threshold : percentage threshold of nan values per feature
    Returns:
        filtered_tx : x matrix without features with higher than threshold percentage of nan's

    """
    num_rows, num_cols = tx.shape
    nan_threshold = threshold * num_rows

    columns_to_keep = []
    for i in range(num_cols):
        nan_count = np.sum(np.isnan(tx[:, i]))
        if nan_count <= nan_threshold:
            columns_to_keep.append(i)

    filtered_tx = tx[:, columns_to_keep]
    return filtered_tx


# For some reason this function will bug when using the standardised data, the output won't be of same size as when using original data


# Returns correlation matrix for value of correlation factor for every value with every other value of dataset
# Diagonal is 1's because correlation of feature with itself
def correlation_matrix(tx):
    """
    This function prodcues correlation matrix of a matrix tx

    Args:
        tx : matrix of dataset
    Returns:
        correlation_matrix : symmetric matrix with correlation between each feature of the matrix with each other

    """
    correlation_matrix = np.corrcoef(np.array(tx), rowvar=False)
    return correlation_matrix


def check_top_right_both(matrix, threshold):
    positions = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            if matrix[i, j] > threshold:
                positions.append((i, j))
    return positions


def check_top_right_one(matrix, threshold):
    """
    This function returns array of the indices of features that have a threshold correlation with another feature of the matrix

    Args:
        tx : matrix of dataset
        threshold : threshold for correlation coefficient between two features
    Returns:
        positions : indices of features that have correlation coefficient superior to threshold to another feature

    """
    positions = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            if matrix[i, j] > threshold:
                positions.append(j)
    return positions


# Remove both correlated features from matrix
def remove_both_correlated(tx, threshold):
    def correlation_matrix(tx):
        correlation_matrix = np.corrcoef(np.array(tx), rowvar=False)
        return correlation_matrix

    corr = correlation_matrix(tx)
    positions = check_top_right_both(corr, threshold)

    matrix_no_correlations = np.delete(tx, positions, axis=1)
    return matrix_no_correlations


# Removes only one of the correlated features
def remove_one_correlated(tx, threshold):
    """
    This function returns array of the indices of features that have a threshold correlation with another feature of the matrix

    Args:
        tx : matrix of dataset
        threshold : threshold for correlation coefficient between two features
    Returns:
        matrix_no_correlation : matrix without features with threhsold correlation coefficient to one another

    """

    def correlation_matrix(tx):
        correlation_matrix = np.corrcoef(np.array(tx), rowvar=False)
        return correlation_matrix

    corr = correlation_matrix(tx)
    positions = check_top_right_one(corr, threshold)

    matrix_no_correlations = np.delete(tx, positions, axis=1)
    return matrix_no_correlations


# Removes features with correlation to y lower than threshold
def remove_low_corr_to_y(tx, y_stand, threshold):
    count = []

    def check_correlation_to_y(data, y_column):
        correlations = []
        for i in range(data.shape[1]):
            feature = data[:, i]
            correlation = np.corrcoef(feature, y_column, rowvar=False)[0, 1]
            correlations.append(correlation)
        return correlations

    for i in range(len(check_correlation_to_y(tx, y_stand))):
        if np.absolute(check_correlation_to_y(tx, y_stand)[i]) >= threshold:
            count.append(i)
    tx_new = tx[:, count]

    return tx_new, count




# Replace nan by mean
def replace_nan_with_mean(feature):
    """
    This function returns array of the indices of features that have a threshold correlation with another feature of the matrix

    Args:
        tx : matrix of dataset
        threshold : threshold for correlation coefficient between two features
    Returns:
        feature :

    """
    feature_mean = np.nanmean(feature)
    feature[np.isnan(feature)] = feature_mean
    return feature


# Replace nan by 0's
def replace_nan_with_zero(feature):
    feature_with_zero = np.nan_to_num(feature, nan=0.0)
    return feature_with_zero


def replace_nan_with_mean_data(data):
    """
    This function returns dataset matrix with nan values replaced by mean of their respective features

    Args:
        tx : matrix of dataset
    Returns:
        data : matrix with nan values replaced by mean of their respective features

    """
    means = np.nanmean(data, axis=0)
    nan_mask = np.isnan(data)
    data[nan_mask] = np.take(means, np.where(nan_mask)[1])
    return data


# Removes features where variance is below certain threshold, ignoring all nan values
def remove_low_variance_features_nonan(x, threshold):
    variances = np.nanvar(x, axis=0)
    new_x = np.copy(x)
    columns_to_keep = variances >= threshold

    new_x = new_x[:, columns_to_keep]
    return new_x


# Removes features where variance is below certain threshold, taking into account nan values -> features with nan values will have variance=nan
def remove_low_variance_features_withnan(x, threshold):
    variances = np.var(x, axis=0)
    new_x = np.copy(x)
    columns_to_keep = variances >= threshold

    new_x = new_x[:, columns_to_keep]
    return new_x


# Checks percentage of 1's in y depending only on singular value for singular feature, for every value of the selected feature
# Returns array of values of the feature as well as the correspondin percentage of 1's in y conditional on this value of the feature
def univariate_correlation(tx, y, feature_to_check):
    column_values = tx[:, feature_to_check]
    unique_values = np.unique(tx[:, feature_to_check])

    percentages = []

    for value in unique_values:
        if np.isnan(value):
            nan_count = np.sum(np.isnan(column_values))
            y_nan_count = sum(1 for j in range(len(tx)) if np.isnan(tx[j, feature_to_check]) and y[j] == 1)
            if nan_count > 0:
                percentage = (y_nan_count / nan_count) * 100
                percentages.append(percentage)
        else:
            total_count = sum(1 for x in column_values if x == value and not np.isnan(x))
            y_test_count = sum(1 for j in range(len(tx)) if
                               tx[j, feature_to_check] == value and y[j] == 1 and not np.isnan(tx[j, feature_to_check]))
            if total_count > 0:
                percentage = (y_test_count / total_count) * 100
                percentages.append(percentage)

    return unique_values, percentages


# Will remove features where corresponding percentage of 1's in y for every value of the feature is close to overall percentage of 1's in y
# "Removes features that have no direct impact on y"
def remove_no_unicorrelation_features(tx, y, delta):
    y_adjusted = np.where(y == -1, 0, y)  # Convert -1s to 0s for calculation
    overall_percentage = (sum(y_adjusted) / len(y_adjusted)) * 100

    for i in range(tx.shape[1] - 1):
        unique_values, percentages = univariate_correlation(tx, y, i)
        k = 0
        for j in range(len(percentages)):
            if percentages[j] - delta <= overall_percentage <= percentages[j] + delta:
                k = k + 1
        if k == len(percentages):
            tx = np.delete(tx, i, axis=1)

    return tx


# Note: when i checked on the full unmodified dataset this removed only 1 feature :(

# Correlation between two features
def correlations_between_2_features(X, Y):
    correlation = np.corrcoef(X, Y)[0, 1]
    return correlation


# Full data clean
def preprocess(tx, threshold_nan, threshold_correlation):
    tx_remove_useless_new, ind_1 = remove_useless_new(tx)
    tx_no8 = eight_answers_to_zero(tx_remove_useless_new)
    tx_no9 = nine_answers_to_nan(tx_no8)
    tx_no7 = seven_answers_to_nan(tx_no9)

    tx_standard = standardize_nan_preserved1(tx_no7)
    tx_high_nanremovestand = remove_high_nan_features(tx_standard, threshold_nan)
    tx_nanmean = replace_nan_with_mean_data(tx_high_nanremovestand)

    tx_nocorr = remove_one_correlated(tx_nanmean, threshold_correlation)

    return tx_nocorr


def preprocess_train(tx, y, threshold_nan, threshold_correlation, threshold_corr_to_y):
    y_stand, junk, junk1 = standardize(y)

    tx_new = preprocess(tx, threshold_nan, threshold_correlation)
    junk_tx, count = remove_low_corr_to_y(tx_new, y_stand, threshold_corr_to_y)

    return junk_tx, np.array(count)


def preprocess_test(tx, threshold_nan, threshold_correlation, count):
    tx_preprocess = preprocess(tx, threshold_nan, threshold_correlation)
    tx_last = tx_preprocess[:, count]

    return tx_last
