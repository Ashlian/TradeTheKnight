import time

from knight_trader import ExchangeClient


SYMBOL = "SPOTDEMO"  # Replace with a real tradable symbol on competition day.
ORDER_SIZE = 1.0
MIN_SPREAD = 0.05
REFRESH_SECS = 0.5


def run():
    client = ExchangeClient()
    resting_bid = None
    resting_ask = None
    next_refresh = 0.0

    try:
        for state in client.stream_state():
            try:
                if state.get("competition_state") != "live":
                    continue
                if time.monotonic() < next_refresh:
                    continue

                book = state.get("book", {}).get(SYMBOL, {})
                bids = sorted((float(price) for price in book.get("bids", {}).keys()), reverse=True)
                asks = sorted(float(price) for price in book.get("asks", {}).keys())

                if not bids or not asks:
                    continue

                best_bid = bids[0]
                best_ask = asks[0]
                spread = best_ask - best_bid

                if spread < MIN_SPREAD:
                    continue

                if resting_bid:
                    client.cancel(resting_bid)
                if resting_ask:
                    client.cancel(resting_ask)

                bid_px = round(best_bid + 0.01, 4)
                ask_px = round(best_ask - 0.01, 4)

                resting_bid = client.buy(SYMBOL, bid_px, ORDER_SIZE)
                resting_ask = client.sell(SYMBOL, ask_px, ORDER_SIZE)
                next_refresh = time.monotonic() + REFRESH_SECS
            except Exception as exc:
                print(f"starter bot error: {exc}")
                time.sleep(1.0)
    finally:
        client.close()


if __name__ == "__main__":
    run()