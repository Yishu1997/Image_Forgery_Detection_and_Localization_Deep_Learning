# Importing libraries

import keras.backend as K 

# Function that calculates True Positives, True Negative, False Positives and False Negatives

def tptnfpfn(y_true, y_pred):
    """
    
    Parameters
    ----------
    y_true : Tensor of Groundtruth masks
    y_pred : Tensor of Predicited masks

    """
    
    # Positive and Negative Predicitions
    
    positive_y_pred =   K.clip(y_pred, 0, 1)
    positive_y_pred = K.round(positive_y_pred)
    negative_y_pred = 1- positive_y_pred
    
    # Positive and Negative Groundtruth
    positive_y = K.clip(y_true, 0 ,1)
    positive_y = K.round(positive_y)
    negative_y = 1 -positive_y
    
    # Calculating True Posotive
    true_positive = K.sum(positive_y * positive_y_pred)
    
    # Calculating True Negative
    true_negative = K.sum(negative_y * negative_y_pred)
    
    # Calculating False Posotive
    false_positive = K.sum(negative_y * positive_y_pred)
    
    # Calculating False Negative
    false_negative = K.sum(positive_y * negative_y_pred)
    
    return true_positive, true_negative, false_positive, false_negative

# Function that calculates the F1 score

def f1_score(y_true, y_pred):
    """
    
    Parameters
    ----------
    y_true : Tensor of Groundtruth masks
    y_pred : Tensor of Predicted masks

    """
    
    true_positive, true_negative, false_positive, false_negative = tptnfpfn(y_true, y_pred)
    
    # Calculating Precision
    precision = true_positive / (true_positive + false_positive + K.epsilon())
    
    # Calculating Recall
    recall = true_positive / (true_positive + false_negative + K.epsilon())
    
    # Calculating F1 score
    f1 = 2 * (precision * recall)  / (precision + recall + K.epsilon())
    
    return f1

# Function to calculate Matthews correlation coefficient

def MCC(y_true, y_pred):
    """
    
    Parameters
    ----------
    y_true : Tensor of Groundtruth masks
    y_pred : Tensor of Predicted masks
    
    """
    
    true_positive, true_negative, false_positive, false_negative = tptnfpfn(y_true, y_pred)
    
    # Numerator = true positive * true negative - false positive * false negative
    
    N = (true_positive * true_negative - false_positive * false_negative)
    
    # Denominator  = square root of {(true positive + false positive) * (true positive + false negative) * (true negative + false positive) * (true negative + fasle negative)}
    
    D = K.sqrt((true_positive + false_positive) * (true_positive + false_negative) * (true_negative + false_positive) * (true_negative + false_negative))
    
    return N / ( D + K.epsilon())
    
# Function to calculate Intersection over Union (IoU)

def IoU(y_true, y_pred):
    """
    
    Parameters
    ----------
    y_true : Tensor of Groundtruth masks
    y_pred : Tensor of Predicted masks

    """
    
    true_positive, true_negative, false_positive, false_negative = tptnfpfn(y_true, y_pred)
    
    # IoU = Mean of (true positive/ (true positive + false positive + false negative))
    
    IoU = K.mean(true_positive / ( true_positive + false_positive + false_negative + K.epsilon()))
    
    return IoU