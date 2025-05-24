from typing import Any, Dict, Iterable, List

import numpy as np
from gluonts.core.component import validated
from gluonts.dataset.common import DataEntry
from gluonts.ev.metrics import \
    MeanWeightedSumQuantileLoss  # Notazione completa per evitare conflitti
from gluonts.ev.metrics import (MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE,
                                SMAPE)
from gluonts.model.forecast import QuantileForecast


def calculate_single_series_metrics(
    forecast: QuantileForecast,
    data_entry: DataEntry,
    train_entry: DataEntry,
    metrics_instances: List[validated],
    seasonality: int,
    prediction_length: int,
    freq: str
) -> Dict[str, float]:
    """
    Calculates metrics for a single forecast/target pair.
    """
    actual_target = data_entry["target"]

    # Find the start index of the forecast in the actual_target array
    # Assume forecast starts right after the end of the train_target
    train_target = train_entry["target"]
    forecast_start_idx_in_actual = len(train_target)

    # Extract the evaluation target segment
    forecast_start_idx_idx_in_actual = forecast_start_idx_in_actual + prediction_length
    eval_target = actual_target[forecast_start_idx_idx_in_actual : forecast_start_idx_in_actual + prediction_length]

    if len(eval_target) != prediction_length:
        # Skip this series if evaluation target length is incorrect
        return {}

    results: Dict[str, float] = {}

    try:
        mean_forecast = forecast.mean if hasattr(forecast, 'mean') else (forecast.quantile(0.5) if hasattr(forecast, 'quantile') else None)
        median_forecast = forecast.quantile(0.5) if hasattr(forecast, 'quantile') else mean_forecast

        if mean_forecast is not None:
            mean_forecast = np.asarray(mean_forecast)
        if median_forecast is not None:
            median_forecast = np.asarray(median_forecast)

        if (mean_forecast is not None and len(mean_forecast) != prediction_length) or \
            (median_forecast is not None and len(median_forecast) != prediction_length and mean_forecast is None):
            return {} # Skip if forecast length is incorrect


        for metric_instance in metrics_instances:
            metric_type = type(metric_instance)
            # metric_key_map = metric_keys.get(metric_type) # Assumi metric_keys è disponibile

            try:
                # Use the actual metric keys map passed or defined globally if needed
                # Assuming metric_keys is globally accessible or passed
                metric_key_map_global = {
                    type(MSE): { "mean": "eval_metrics/MSE[mean]", 0.5: "eval_metrics/MSE[0.5]", },
                    type(MAE): { 0.5: "eval_metrics/MAE[0.5]", },
                    type(MASE): "eval_metrics/MASE[0.5]",
                    type(MAPE): "eval_metrics/MAPE[0.5]",
                    type(SMAPE): "eval_metrics/sMAPE[0.5]",
                    type(MSIS): "eval_metrics/MSIS",
                    type(RMSE): { "mean": "eval_metrics/RMSE[mean]", },
                    type(NRMSE): { "mean": "eval_metrics/NRMSE[mean]", },
                    type(ND): { 0.5: "eval_metrics/ND[0.5]", },
                    type(MeanWeightedSumQuantileLoss): "eval_metrics/mean_weighted_sum_quantile_loss",
                }
                metric_key_map = metric_key_map_global.get(metric_type)

                if isinstance(metric_instance, (MSE, MAE, RMSE, NRMSE, ND, MAPE, SMAPE)):
                    forecast_val = mean_forecast if metric_instance.forecast_type == "mean" else median_forecast
                    if forecast_val is not None:
                        val = metric_instance(eval_target, forecast_val)
                        if isinstance(metric_key_map, dict):
                            key = metric_key_map.get(metric_instance.forecast_type)
                            if key: results[key] = float(val)

                elif isinstance(metric_instance, MASE):
                    if median_forecast is not None:
                        val = metric_instance(eval_target, median_forecast, train_target, seasonality)
                        if isinstance(metric_key_map, str):
                            results[metric_key_map] = float(val)

                elif isinstance(metric_instance, MSIS):
                    if hasattr(forecast, 'quantile') and median_forecast is not None:
                        alpha = 0.05
                        lower_q = forecast.quantile(alpha / 2) if hasattr(forecast, 'quantile') else None
                        upper_q = forecast.quantile(1 - alpha / 2) if hasattr(forecast, 'quantile') else None

                        if lower_q is not None and upper_q is not None and median_forecast is not None:
                            lower_q = np.asarray(lower_q)
                            upper_q = np.asarray(upper_q)
                            if len(lower_q) == prediction_length and len(upper_q) == prediction_length:
                                val = metric_instance(eval_target, median_forecast, lower_q, upper_q, train_target, seasonality)
                                if isinstance(metric_key_map, str):
                                    results[metric_key_map] = float(val)

                elif isinstance(metric_instance, MeanWeightedSumQuantileLoss):
                    if hasattr(forecast, 'quantile'):
                        quantile_levels = metric_instance.quantile_levels
                        quantiles_forecasted = {q: np.asarray(forecast.quantile(q)) for q in quantile_levels if hasattr(forecast, 'quantile') and np.asarray(forecast.quantile(q)).shape[-1] == prediction_length} # Ensure correct shape
                        if len(quantiles_forecasted) == len(quantile_levels): # Ensure all required quantiles were obtained
                            val = metric_instance(eval_target, quantiles_forecasted)
                            if isinstance(metric_key_map, str):
                                results[metric_key_map] = float(val)
            except Exception as metric_e:
                # Optionally print metric-specific error
                pass
    except Exception as e:
        return {}

    return results

def evaluate_forecasts(
    forecasts_iterable: Iterable[QuantileForecast],
    test_data_iterable: Iterable[DataEntry],
    train_data_iterable: Iterable[DataEntry],
    metrics_instances: List[validated],
    seasonality: int,
    prediction_length: int,
    freq: str # Pass frequency
) -> Dict[str, Any]:
    """
    Manually calculates evaluation metrics by iterating over forecasts and test data.
    """
    all_series_metrics: List[Dict[str, float]] = []

    try:
        # Load train data into list for MASE calculation. Memory intensive.
        train_data_list = list(train_data_iterable)

        for i, (forecast, test_entry) in enumerate(zip(forecasts_iterable, test_data_iterable)):
            if i < len(train_data_list):
                train_entry = train_data_list[i]
            else:
                continue # Skip this series

            # Ensure required data entry fields are available or added if missing
            if 'freq' not in test_entry:
                test_entry['freq'] = freq # Add frequency if not present

            # Calculate metrics for the single series
            series_metrics = calculate_single_series_metrics(
                forecast=forecast,
                data_entry=test_entry,
                train_entry=train_entry,
                metrics_instances=metrics_instances,
                seasonality=seasonality,
                prediction_length=prediction_length,
                freq=freq # Pass frequency
            )
            if series_metrics:
                all_series_metrics.append(series_metrics)

    except Exception as e:
        pass


    aggregated_results: Dict[str, float] = {}
    if not all_series_metrics:
        expected_keys = {}
        metric_key_map_global = {
            type(MSE): { "mean": "eval_metrics/MSE[mean]", 0.5: "eval_metrics/MSE[0.5]", },
            type(MAE): { 0.5: "eval_metrics/MAE[0.5]", },
            type(MASE): "eval_metrics/MASE[0.5]",
            type(MAPE): "eval_metrics/MAPE[0.5]",
            type(SMAPE): "eval_metrics/sMAPE[0.5]",
            type(MSIS): "eval_metrics/MSIS",
            type(RMSE): { "mean": "eval_metrics/RMSE[mean]", },
            type(NRMSE): { "mean": "eval_metrics/NRMSE[mean]", },
            type(ND): { 0.5: "eval_metrics/ND[0.5]", },
            type(MeanWeightedSumQuantileLoss): "eval_metrics/mean_weighted_sum_quantile_loss",
        }
        for metric_instance in metrics_instances:
            metric_type = type(metric_instance)
            keys = metric_key_map_global.get(metric_type)
            if isinstance(keys, dict):
                expected_keys.update(keys)
            elif isinstance(keys, str):
                expected_keys[keys] = None # Value doesn't matter here

        for key in expected_keys.keys():
            aggregated_results[key] = float('nan') # Use NaN for missing results
        return aggregated_results


    sum_metrics: Dict[str, float] = {}
    for series_metrics in all_series_metrics:
        for key, value in series_metrics.items():
            sum_metrics[key] = sum_metrics.get(key, 0.0) + value

    count_metrics: Dict[str, int] = {}
    for series_metrics in all_series_metrics:
        for key in series_metrics.keys():
            count_metrics[key] = count_metrics.get(key, 0) + 1


    for key, total_sum in sum_metrics.items():
        count = count_metrics.get(key, 0)
        if count > 0:
            aggregated_results[key] = total_sum / count
        else:
            aggregated_results[key] = float('nan')

    return aggregated_results