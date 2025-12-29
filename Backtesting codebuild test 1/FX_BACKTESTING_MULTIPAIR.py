import MetaTrader5 as mt5
from backtesting import Backtest, Strategy
import pandas as pd
from datetime import datetime
import pytz

# Function to calculate SL and TP prices
def calculate_prices(entry_price, risk_reward_ratio, order_type, mean_candle_size):
    risk_part, reward_part = map(int, risk_reward_ratio.split(':'))
    risk_amount = mean_candle_size * risk_part
    reward_amount = mean_candle_size * reward_part
    
    if order_type == 'buy':
        sl_price = entry_price - risk_amount
        tp_price = entry_price + reward_amount
    elif order_type == 'sell':
        sl_price = entry_price + risk_amount
        tp_price = entry_price - reward_amount
    else:
        raise ValueError("order_type must be either 'buy' or 'sell'")
    
    return sl_price, tp_price

# Set timezone to UTC
utc_tz = pytz.utc

# Initialize MT5 connection
if not mt5.initialize():
    print("Failed to initialize MT5")
    exit()

login = 51708234
password = "4bM&wuVJcBTnjV"
server = "ICMarketsEU-Demo"
if not mt5.login(login, password, server):
    print("Failed to login to MT5")
    mt5.shutdown()
    exit()

# Define currency pairs with their respective mean candle sizes and risk-reward ratios
currency_pairs = {
    "USDCAD": {"mean_candle_size": 0.0088, "risk_reward_ratiosell": "3:4", "risk_reward_ratiobuy": "2:3"},
    "GBPUSD": {"mean_candle_size": 0.0102, "risk_reward_ratiosell": "1:2", "risk_reward_ratiobuy": "1:2"}
}

timeframe = mt5.TIMEFRAME_D1

# Define start date and end date
start_date = datetime(2023, 2, 2, 0, 0, 0, tzinfo=utc_tz)
end_date = datetime(2024, 6, 22, 0, 0, 0, tzinfo=utc_tz)

for symbol, params in currency_pairs.items():
    # Retrieve OHLC data from MetaTrader
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None:
        print(f"Failed to retrieve data for {symbol}")
        continue

    ohlc_data = pd.DataFrame(rates)
    ohlc_data['time'] = pd.to_datetime(ohlc_data['time'], unit='s')
    ohlc_data.set_index('time', inplace=True)

    # Select only the required columns
    df = ohlc_data[['open', 'high', 'low', 'close']]
    df.columns = ['Open', 'High', 'Low', 'Close']

    # Load predictions from CSV files
    df_predsell = pd.read_csv(f'predict{symbol}_D1Sell.csv', index_col=0)
    df_predsell.index = pd.to_datetime(df_predsell.index)

    df_predbuy = pd.read_csv(f'predict{symbol}_D1Buy.csv', index_col=0)
    df_predbuy.index = pd.to_datetime(df_predbuy.index)

    # Define a simple strategy based on predictions
    class SimpleStrategy(Strategy):
        mean_candle_size = params["mean_candle_size"]
        risk_reward_ratiosell = params["risk_reward_ratiosell"]
        risk_reward_ratiobuy = params["risk_reward_ratiobuy"]

        def init(self):
            self.mean_candle_size = SimpleStrategy.mean_candle_size
            self.risk_reward_ratiosell = SimpleStrategy.risk_reward_ratiosell
            self.risk_reward_ratiobuy = SimpleStrategy.risk_reward_ratiobuy

        def next(self):
            current_time = self.data.index[-1]
            
            # Get predictions for the current time
            sell_prediction = df_predsell.loc[current_time, 'prediction'] if current_time in df_predsell.index else 0
            buy_prediction = df_predbuy.loc[current_time, 'prediction'] if current_time in df_predbuy.index else 0
            
            # Ensure no trade is placed if both predictions are 1
            if sell_prediction == 1 and buy_prediction == 1:
                return
            
            # Place a sell order if the sell prediction is 1
            if sell_prediction == 1 and buy_prediction != 1:
                entry_price = self.data.Close[-1]
                order_type = 'sell'
                sl_price, tp_price = calculate_prices(entry_price, self.risk_reward_ratiosell, order_type, self.mean_candle_size)
                self.sell(size=100000, sl=sl_price, tp=tp_price)
            
            # Place a buy order if the buy prediction is 1
            if buy_prediction == 1 and sell_prediction != 1:
                entry_price = self.data.Close[-1]
                order_type = 'buy'
                sl_price, tp_price = calculate_prices(entry_price, self.risk_reward_ratiobuy, order_type, self.mean_candle_size)
                self.buy(size=100000, sl=sl_price, tp=tp_price)

    # Create and run backtest with the SimpleStrategy
    bt = Backtest(df, SimpleStrategy, cash=10000, commission=0.0003, margin=0.01)
    output = bt.run()
    print(f"Results for {symbol}:")
    print(output)

    risk_reward_ratios = ['1:1', '1:2', '1:3', '1:4', '2:2', '2:3', '2:4', '2:5']

    # Optimize the strategy for SL/TP ratios
    stats = bt.optimize(
        risk_reward_ratiosell=risk_reward_ratios,
        risk_reward_ratiobuy=risk_reward_ratios,
        maximize='Equity Final [$]',
        constraint=lambda p: ':' in p.risk_reward_ratiosell and ':' in p.risk_reward_ratiobuy
    )

    # Print optimized strategy stats
    print(f"Best risk_reward_ratiosell for {symbol}: {stats._strategy.risk_reward_ratiosell}")
    print(f"Best risk_reward_ratiobuy for {symbol}: {stats._strategy.risk_reward_ratiobuy}")
    print(stats)

    # Plot the results of the optimized strategy
    bt.plot()

# Shutdown MT5 connection after data retrieval
mt5.shutdown()
