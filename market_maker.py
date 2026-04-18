import time
import numpy as np
from knight_trader import ExchangeClient

SYMBOL = "SPOTDEMO"  # Replace with a real tradable symbol on competition day.
ORDER_SIZE = 1.0
MIN_SPREAD = 0.05
REFRESH_SECS = 0.5

def get_symbol_trades(state):
    return [t for t in state.get("trades") if t.get("symbol") == SYMBOL]

def run():
    client = ExchangeClient()
    #portfolio keep a dictionary of ids with values -1 or 1 depending on buy or ask
    portfolio = NONE
    position = 0.0
    next_refresh = 0.0

    try:
        for state in client.stream_state():
            try:
                if state.get("competition_state") != "live":
                    continue
                if time.monotonic < next_refresh:
                    continue
                book = state.get("book", {}).get(SYMBOL, {})
                #check if id in portfolio is in the order book, if its not then change position. Check if each key is in the order book

                bids = sorted((float(price) for price in book.get("bids", {}).keys()), reverse=True)
                asks = sorted(float(price) for price in book.get("asks", {}).keys())
                trades = state.get("trades", {})

                '''
                grab best bid/best ask, compute spread/mid, if spread < min_spread, then stop otherwise, compute imbalance/skew, structural vol and reservatin price

                set orders at reservation price +- half spread
                append order ids
                '''

                if not bids or not asks:
                    continue
                timeseries = NONE
                
            except Exception as exc:
                print(f"bot error: {exc}")
                time.sleep(1.0)
    
    finally:
        client.close()

if __name__ == "__main__":
    run()