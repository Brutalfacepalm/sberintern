from model import get_scaler, get_model
import re
from utils import load_data_for_forecast, get_min_stab_value_for_period, get_argparse


if __name__ == "__main__":
    parser = get_argparse()
    args = parser.parse_args()
    month_pred = int(re.findall(r'[0-9]+', args.month_pred)[0])

    df_forecast, ru_holidays = load_data_for_forecast(args.input_file, args.split_date)
    model = get_model(args.path_model)
    scaler = get_scaler(args.path_scaler)

    y_pred_min = get_min_stab_value_for_period(df_forecast, model, scaler, ru_holidays,
                                               month_pred=month_pred, stack=args.stack,
                                               masked=args.masked, asc_mask=args.asc_mask,
                                               smooth=args.smooth, scale_coef=args.scale_coef)

    print(f'The actual stable part: {y_pred_min}')
