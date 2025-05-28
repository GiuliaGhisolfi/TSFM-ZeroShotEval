from gluonts.ev.metrics import (MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE,
                                SMAPE, MeanWeightedSumQuantileLoss)

# Instantiate the metrics
metrics = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(
        quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ),
]

def get_metrics():
    """
    Returns the list of metrics to be used for evaluation.
    
    Returns:
        list: A list of instantiated metric objects.
    """
    return metrics

def evaluate_metrics(forecast, target):
    """
    Evaluates the provided forecast against the target using the defined metrics.
    
    Args:
        forecast: The forecast object containing predictions.
        target: The target object containing actual values.
    
    Returns:
        dict: A dictionary with metric names as keys and their computed values.
    """
    metrics = get_metrics()
    results = {metric: metric(forecast, target) for metric in metrics}
    return results