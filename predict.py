def predict_flower(clf, input_features):
    """
    Predict iris type

    Args:
        clf (sklearn estimator): a trained classifier 
        input_features (np.array like): 4 numeric values
    Returns:
        np.arry of string dtype: the model's prediction
    
    """
    return clf.predict(input_features)