from os.path import exists

import argparse
import pandas as pd
import numpy as np
import holidays
import torch
import gdown


DEL = ';'
DATES_COLUMN = 'REPORTDATE'
VALUES_COLUMN = 'VALUE'
MAX_SEQ_LEN = 91
DEEP_TIME = 31
DAYS_MONTH = 30
ZOOM = 1e9


def get_argparse():
    parser = argparse.ArgumentParser(description='More information about research read there: '
                                                 'https://github.com/Brutalfacepalm/sberintern'
                                                 'Research this pytorch model read there: '
    'https://colab.research.google.com/drive/1UmVPYV3gCJUPoO-uTlLq3FpMClTzlblx#scrollTo=CHb3WbDTjLXI&uniqifier=1')
    parser.add_argument('-i', '--input_file', dest='input_file',
                        type=str, default='data.csv', nargs='?',
                        help='Name of file with data in CSV format. Default: date.csv')

    parser.add_argument('-d', '--date', dest='split_date',
                        type=str, default='2019-02-01', nargs='?',
                        help='Start date forecast. Need valid Date format. Default: 2019-02-01')
    parser.add_argument('-mp', '--month_pred', dest='month_pred',
                        type=str, default='1M', nargs='?',
                        help='Months forecast. Maybe 1M-12M. Default: 1M')

    parser.add_argument('-mdl', '--path_model', dest='path_model',
                        type=str, default='model_state', nargs='?',
                        help='File state of model. Default: model_state')
    parser.add_argument('-scl', '--path_scaler', dest='path_scaler',
                        type=str, default='scaler_state', nargs='?',
                        help='File state of scaler. Default: scaler_state')

    parser.add_argument('--stack', dest='stack',
                        type=bool, default=True, nargs='?',
                        help='Stacked forecasting. Default: True')
    parser.add_argument('--masked', dest='masked',
                        type=str, default='mask', nargs='?',
                        help='Masked method when stacked. Maybe mean, mask, both. Default: mask')
    parser.add_argument('--asc_mask', dest='asc_mask',
                        type=bool, default=True, nargs='?',
                        help='Sort mask weights ascending or descending. Default: True')
    parser.add_argument('--smooth', dest='smooth',
                        type=bool, default=True, nargs='?',
                        help='Smoothing prediction values on ewm method with span=3. Default: True')
    parser.add_argument('--scale_coef', dest='scale_coef',
                        type=float, default=0.9, nargs='?',
                        help='Scale coefficient for multiply result value. Default: 0.9')

    return parser


def mask(length, ascending=True):
    x = np.array([np.log(s)*(s-1) for s in range(length+1, 0, -1)][:-1])
    if ascending:
        return x/sum(x)
    elif not ascending:
        return x[::-1]/sum(x)


def generate_feature(df, ru_holidays):
    df['day_num'] = df.REPORTDATE.dt.day
    df['day_name'] = df.REPORTDATE.dt.day_name()
    df['day_of_year'] = df.REPORTDATE.dt.day_of_year
    df['year'] = df.REPORTDATE.dt.isocalendar().year
    df['month'] = df.REPORTDATE.dt.month
    df['week'] = df.REPORTDATE.dt.isocalendar().week
    df['is_holiday'] = df.REPORTDATE.isin(ru_holidays)

    slc = 7
    holidays_7 = []
    if df.shape[0] >= slc:
        while slc > 1:
            for i, v in enumerate(df['is_holiday']):
                if df['is_holiday'].shape[0] < i + slc:
                    slc -= 1
                if df['is_holiday'].iloc[i:i + slc].any():
                    holidays_7.append('True')
                else:
                    holidays_7.append('False')
        df['holidays_7'] = holidays_7
    else:
        df['holidays_7'] = False

    for p in [1, 2, 3, 5, 7, 10, 14, 21, 31, 61, 91]:
        df[f'diff_value_{p}'] = df['VALUE'].diff(periods=p)
        df[f'diff_value_percent_{p}'] = df[f'diff_value_{p}'] / df['VALUE'] * 100

    for i in [14, 21, 28, 35, 42, 49]:
        df[f'ma_mean_{i}'] = df.VALUE.rolling(i).mean()
        df[f'ema_mean_{i}'] = df.VALUE.ewm(span=7, adjust=False).mean()
        df[f'ma_std_{i}'] = df.VALUE.rolling(i).std()

    for min_period_trend in [14, 21, 28, 35, 42, 49]:
        mean_values = []
        std_values = []
        diff_values = []

        for i, _ in enumerate(df.VALUE):
            idx_start = i // min_period_trend * min_period_trend
            idx_end = (i // min_period_trend + 1) * min_period_trend

            mean_values.append(df.VALUE.iloc[idx_start: idx_end].mean())
            std_values.append(df.VALUE.iloc[idx_start: idx_end].std())

            if i < min_period_trend:
                diff_values.append(0)
            else:
                diff_values.append(mean_values[-1] - mean_values[-min_period_trend - 1])

        merge_mean_values = []
        trend_mean_values = {df.REPORTDATE.iloc[0]: df.VALUE.iloc[0]}
        trend_values = []

        diff_idx_start = 0
        diff_idx_end = 1

        while diff_idx_end < len(diff_values):
            if np.sign(diff_values[diff_idx_start]) == np.sign(diff_values[diff_idx_end]):
                diff_idx_end += 1
            else:
                merge_mean_values.extend(
                    [np.mean(mean_values[diff_idx_start: diff_idx_end])] * (diff_idx_end - diff_idx_start))

                end_trend = mean_values[diff_idx_end - 1]
                trend_mean_values[df.REPORTDATE.iloc[diff_idx_end - 1]] = end_trend
                step_trend = (mean_values[diff_idx_end - 1] -
                              mean_values[diff_idx_start]) / \
                             (diff_idx_end - diff_idx_start)
                trend_values.append(mean_values[diff_idx_start])
                trend_values.extend(
                    [mean_values[diff_idx_start] + step_trend * i for i in range(1, diff_idx_end - diff_idx_start)])

                diff_idx_start = diff_idx_end
                diff_idx_end += 1

            if diff_idx_end == len(diff_values):
                merge_mean_values.extend(
                    [np.mean(mean_values[diff_idx_start: diff_idx_end])] * (diff_idx_end - diff_idx_start))

                end_trend = mean_values[diff_idx_end - 1]
                trend_mean_values[df.REPORTDATE.iloc[-1]] = end_trend
                step_trend = (mean_values[diff_idx_end - 1] -
                              mean_values[diff_idx_start]) / \
                             (diff_idx_end - diff_idx_start)
                trend_values.append(mean_values[diff_idx_start])
                trend_values.extend(
                    [mean_values[diff_idx_start] + step_trend * i for i in range(1, diff_idx_end - diff_idx_start)])

                break

        df[f'mean_subtrend_{min_period_trend}'] = mean_values
        df[f'trend_values_{min_period_trend}'] = trend_values
        df[f'diff_trend_values_{min_period_trend}'] = df.VALUE - trend_values

    df = pd.get_dummies(df, columns=['day_name'], prefix='day_is_')

    df = pd.get_dummies(df, columns=['is_holiday'], prefix='is_holiday_')
    if 'is_holiday__True' not in df.columns:
        df['is_holiday__True'] = 0

    df = pd.get_dummies(df, columns=['holidays_7'], prefix='holidays_7_')
    if 'holidays_7__True' not in df.columns:
        df['holidays_7__True'] = 0

    return df


def period_forecast(df_base, model, scaler, ru_holidays, month_pred,
                    stack=False, masked='mean', asc_mask=True):

    last_date = df_base[DATES_COLUMN].iloc[-1]

    forecast_horizont = {}
    with torch.no_grad():
        model.eval()
        if not stack:
            _X_ = generate_feature(df_base, ru_holidays).drop(columns=[DATES_COLUMN]).dropna()
            _X_ = scaler.transform(_X_)

            for i in range(DAYS_MONTH):
                split_idx_start = -DEEP_TIME - DAYS_MONTH + i
                split_idx_end = -DAYS_MONTH + i

                x_test = _X_[split_idx_start:split_idx_end, :]
                x_test = torch.tensor(x_test).float().unsqueeze(0).to(model.device)

                forecast_value = model(x_test).squeeze(0)
                for d, vlue in enumerate(forecast_value):
                    forecast_horizont[
                        last_date + pd.Timedelta(value=DAYS_MONTH * d + i + 1,
                                                 unit='D')] = vlue.item() * ZOOM

            forecast_horizont = {k: v for k, v in sorted(forecast_horizont.items(), key=lambda x: x[0])}
            forecast_horizont = pd.DataFrame.from_dict(forecast_horizont, orient='index').reset_index()
            forecast_horizont.columns = [DATES_COLUMN, VALUES_COLUMN]

            return forecast_horizont

        elif stack:
            df_stack = df_base.copy()

            for idx_stack in range(month_pred):
                _X_ = generate_feature(df_stack, ru_holidays).drop(columns=[DATES_COLUMN]).dropna()
                _X_ = scaler.transform(_X_)

                for i in range(DAYS_MONTH):
                    split_idx_start = -DEEP_TIME - DAYS_MONTH + i
                    split_idx_end = -DAYS_MONTH + i

                    x_test = _X_[split_idx_start:split_idx_end, :]
                    x_test = torch.tensor(x_test).float().unsqueeze(0).to(model.device)

                    forecast_value = model(x_test).squeeze(0)

                    for d, vlue in enumerate(forecast_value):
                        if d + idx_stack < month_pred:
                            if (last_date + pd.Timedelta(value=DAYS_MONTH * (d + idx_stack) + i + 1,
                                                         unit='D')) in forecast_horizont:
                                forecast_horizont[last_date + pd.Timedelta(value=DAYS_MONTH * (d + idx_stack) + i + 1,
                                                                           unit='D')].append(vlue.item() * ZOOM)
                            else:
                                forecast_horizont[
                                    last_date + pd.Timedelta(value=DAYS_MONTH * (d + idx_stack) + i + 1,
                                                             unit='D')] = [vlue.item() * ZOOM]

                if masked == 'mean':
                    forecast_horizont_stack = {k: np.mean(v) for k, v in
                                               sorted(forecast_horizont.items(), key=lambda x: x[0])}
                elif masked == 'mask':
                    forecast_horizont_stack = {k: np.sum(np.array(v) * mask(len(v), ascending=asc_mask)) for k, v in
                                               sorted(forecast_horizont.items(), key=lambda x: x[0])}
                elif masked == 'both':
                    forecast_horizont_stack = {
                        k: (np.sum(np.array(v) * mask(len(v), ascending=asc_mask)) + np.mean(v)) / 2 for k, v in
                        sorted(forecast_horizont.items(), key=lambda x: x[0])}

                forecast_horizont_stack = pd.DataFrame.from_dict(forecast_horizont_stack, orient='index').reset_index()
                forecast_horizont_stack.columns = [DATES_COLUMN, VALUES_COLUMN]

                df_stack = pd.concat(
                    [df_stack, forecast_horizont_stack.iloc[DAYS_MONTH * idx_stack:DAYS_MONTH * (idx_stack + 1), :]],
                    axis=0, ignore_index=True)

                df_stack = df_stack.iloc[-MAX_SEQ_LEN - DEEP_TIME - DAYS_MONTH:, :][
                    [DATES_COLUMN, VALUES_COLUMN]].reset_index(drop=True)

            if masked == 'mean':
                forecast_horizont = {k: np.mean(v) for k, v in sorted(forecast_horizont.items(), key=lambda x: x[0])}
            elif masked == 'mask':
                forecast_horizont = {k: np.sum(np.array(v) * mask(len(v), ascending=asc_mask)) for k, v in
                                     sorted(forecast_horizont.items(), key=lambda x: x[0])}
            elif masked == 'both':
                forecast_horizont = {k: (np.sum(np.array(v) * mask(len(v), ascending=asc_mask)) + np.mean(v)) / 2 for
                                     k, v in sorted(forecast_horizont.items(), key=lambda x: x[0])}

            forecast_horizont = pd.DataFrame.from_dict(forecast_horizont, orient='index').reset_index()
            forecast_horizont.columns = [DATES_COLUMN, VALUES_COLUMN]

            return forecast_horizont


def load_data_for_forecast(path, split_date):
    df_forecast = pd.read_csv(path, delimiter=DEL, parse_dates=[DATES_COLUMN], dayfirst=True)

    ru_holidays = holidays.RU(years=range(df_forecast[DATES_COLUMN].dt.year.min(),
                                          df_forecast[DATES_COLUMN].dt.year.max()+1))
    df_forecast = generate_feature(df_forecast, ru_holidays)
    df_forecast.dropna(inplace=True)

    assert df_forecast[DATES_COLUMN].max() >= (pd.Timestamp(split_date) - pd.Timedelta(value=1, unit='D')), \
        'Not enough data for forecasting. Max date in history must be less than requested by 1 day.'

    df_forecast = df_forecast[df_forecast[DATES_COLUMN] < split_date]

    assert df_forecast.shape[0] >= MAX_SEQ_LEN+DEEP_TIME+DAYS_MONTH, \
        'Not enough data for forecasting. Days in history must be more or equal than 152 days.'

    df_forecast = df_forecast.iloc[-MAX_SEQ_LEN-DEEP_TIME-DAYS_MONTH:, :][[DATES_COLUMN, VALUES_COLUMN]]
    df_forecast = df_forecast.reset_index(drop=True)
    return df_forecast, ru_holidays


def get_ext_ewm(data, pred, span=3):
    pred_ext = pd.concat([data.iloc[-span + 1:, :].reset_index(drop=True)[[DATES_COLUMN, VALUES_COLUMN]], pred],
                         axis=0,
                         ignore_index=True)
    pred_ext[VALUES_COLUMN] = pred_ext[VALUES_COLUMN].ewm(span=span, min_periods=span, adjust=True).mean()
    pred_ext.dropna(inplace=True)
    pred_ext = pred_ext.reset_index(drop=True)
    return pred_ext


def get_min_stab_value_for_period(data, model, scaler, ru_holidays, month_pred,
                                  stack, masked, asc_mask, smooth, scale_coef):

    y_pred = period_forecast(data, model, scaler, ru_holidays, month_pred=month_pred,
                             stack=stack, masked=masked, asc_mask=asc_mask)
    if smooth:
        y_pred_ext = get_ext_ewm(data, y_pred)
        return int((y_pred_ext[VALUES_COLUMN].min())*scale_coef)
    else:
        return int((y_pred[VALUES_COLUMN].min())*scale_coef)


def download_file_from_google_drive(id, destination):
    url = f'https://drive.google.com/uc?id={id}'
    gdown.download(url, destination, quiet=False)


def get_file(path):
    return exists(path)


if __name__ == "__main__":
    pass