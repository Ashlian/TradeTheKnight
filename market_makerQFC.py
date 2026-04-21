import time
import numpy as np
from knight_trader import ExchangeClient

SYMBOL = "PASS"  # Replace with a real tradable symbol on competition day.
ORDER_SIZE = 1.0
MIN_SPREAD = 0.05
REFRESH_SECS = 0.5
GAMMA = 0.25
BASE_SPREAD = 0.3
MAX_INVENTORY = 25.0
DEPTH_LEVELS = 3
THINNESS_LEVELS = 2
GAP_LEVELS = 5
REVERSION_AGGRESSIVENESS = 1.5

W_SPREAD = 0.50
W_THIN = 0.30
W_GAP = 0.20

THIN_SCALE = 0.001
GAP_SCALE = 0.01
IMB_SKEW = 0.02
VOL_SPREAD_MULT = 4.0

def extract_qty(val) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, (list, tuple)):
        total = 0.0
        for item in val:
            if isinstance(item, (int, float)):
                total += float(item)
            elif isinstance(item, dict):
                for key in ("quantity", "qty", "size", "amount", "volume"):
                    if key in item:
                        total += float(item[key])
                        break
        return total
    if isinstance(val, dict):
        for key in ("quantity", "qty", "size", "amount", "volume"):
            if key in val:
                return float(val[key])
    return 0.0

def structural_volatility(bids: dict, asks: dict, mid: float) -> float:
    """
    Estimate volatility purely from current book geometry.
    Returns a float in roughly the same units as a returns-based sigma.
    """

    bid_prices = sorted((float(p) for p in bids), reverse=True)
    ask_prices = sorted(float(p) for p in asks)

    best_bid = bid_prices[0]
    best_ask = ask_prices[0]

    # ── component 1: normalized spread ──────────────────────────────────────
    # wider spread → more uncertainty
    sigma_spread = (best_ask - best_bid) / mid

    # ── component 2: near-touch thinness ────────────────────────────────────
    # low quantity near the top → price moves easily → high fragility
    top_bid_qty = sum(
        extract_qty(bids[str(p)]) if str(p) in bids else 0.0
        for p in bid_prices[:THINNESS_LEVELS]
    )
    top_ask_qty = sum(
        extract_qty(asks[str(p)])if str(p) in asks else 0.0
        for p in ask_prices[:THINNESS_LEVELS]
    )
    total_near_qty = top_bid_qty + top_ask_qty

    # avoid division by zero; more qty = less fragility
    sigma_thin = THIN_SCALE / max(total_near_qty, 0.001)

    # ── component 3: average level spacing ──────────────────────────────────
    # large gaps between price levels → disagreement → uncertainty
    bid_gaps = [
        bid_prices[i] - bid_prices[i + 1]
        for i in range(min(GAP_LEVELS, len(bid_prices) - 1))
    ]
    ask_gaps = [
        ask_prices[i + 1] - ask_prices[i]
        for i in range(min(GAP_LEVELS, len(ask_prices) - 1))
    ]
    all_gaps = bid_gaps + ask_gaps
    avg_gap  = float(np.mean(all_gaps)) if all_gaps else 0.0

    sigma_gap = avg_gap * GAP_SCALE

    # ── weighted composite ───────────────────────────────────────────────────
    sigma = (
        W_SPREAD * sigma_spread +
        W_THIN   * sigma_thin   +
        W_GAP    * sigma_gap
    )

    return max(sigma, 1e-4)  # floor to avoid degenerate quotes

def book_imbalance(bids: dict, asks: dict) -> float:
    """
    Ratio of bid depth to total near-touch depth.
    Returns 0 to 1. Above 0.5 = buying pressure.
    """
    bid_prices = sorted((float(p) for p in bids), reverse=True)
    ask_prices = sorted(float(p) for p in asks)

    bid_qty = sum(
        extract_qty(bids[str(p)]) if str(p) in bids else 0.0
        for p in bid_prices[:DEPTH_LEVELS]
    )
    ask_qty = sum(
        extract_qty(asks[str(p)]) if str(p) in asks else 0.0
        for p in ask_prices[:DEPTH_LEVELS]
    )

    total = bid_qty + ask_qty
    return bid_qty / total if total > 0 else 0.5

def run():
    client = ExchangeClient()
    inventory = 0.0
    bid_id = None
    ask_id = None
    next_refresh = 0.0

    

    try:
        for state in client.stream_state():
            try:
                if state.get("competition_state") != "live":
                    continue
                if time.monotonic() < next_refresh:
                    continue

                book = state.get("book", {}).get(SYMBOL, {})
                bid_dict = book.get("bids", {})
                ask_dict = book.get("asks", {})
                # trades = state.get("trades", {})
                #check if id in portfolio is in the order book, if its not then change position. Check if each key is in the order book
                bid_prices = sorted((float(price) for price in book.get("bids", {}).keys()), reverse=True)
                ask_prices = sorted(float(price) for price in book.get("asks", {}).keys())
                if not bid_prices or not ask_prices:
                    continue

                '''
                grab best bid/best ask, compute spread/mid, if spread < min_spread, then stop otherwise, compute imbalance/skew, structural vol and reservatin price

                set orders at reservation price +- half spread
                append order ids
                '''
                # calculations  
                best_bid = bid_prices[0]
                best_ask = ask_prices[0]
                # best_bid = client.get_best_bid(SYMBOL)
                # best_ask = client.get_best_ask(SYMBOL)
                spread = best_ask - best_bid
                midprice = (best_bid + best_ask) / 2

                if spread < MIN_SPREAD:
                    try: 
                        client.cancel_all()
                    except Exception:
                        pass
                    bid_id = None
                    ask_id = None
                
                sigma = structural_volatility(bid_dict, ask_dict, midprice)

                imbalance = book_imbalance(bid_dict, ask_dict)
                imb_skew = (imbalance - 0.5) * IMB_SKEW
                
                inventory_skew = ((best_ask-best_bid) * REVERSION_AGGRESSIVENESS / MAX_INVENTORY)

                reservation = midprice - GAMMA * (sigma ** 2) * inventory + imb_skew

                half_spread =  max(BASE_SPREAD, VOL_SPREAD_MULT * sigma) / 2

                bid_px = round(reservation - half_spread, 2)
                ask_px = round(reservation + half_spread, 2)

                if bid_px >= ask_px:
                    try:
                        client.cancel_all()
                    except Exception:
                        pass
                    continue
                # bid_px = min(bid_px, best_bid + 0.01)
                # ask_px = max(ask_px, best_ask - 0.01)

                post_bid = inventory < MAX_INVENTORY
                post_ask = inventory > -MAX_INVENTORY

                try: 
                    client.cancel_all()
                except Exception:
                    pass
                bid_id = None
                ask_id = None
                
                # time.sleep(0.05)

                # if bid_id:
                #     client.cancel(bid_id)
                #     bid_id = None
                # if ask_id:
                #     client.cancel(ask_id)
                #     ask_id = None

                if post_bid:
                    bid_id = client.buy(SYMBOL, bid_px, ORDER_SIZE)
                if post_ask:
                    ask_id = client.sell(SYMBOL, ask_px, ORDER_SIZE)
                
                team = client.get_team_state()
                inventory = team.get("inventory", {}).get(SYMBOL, 0)
                next_refresh = time.monotonic() + REFRESH_SECS
            except Exception as exc:
                print(f"bot error: {exc}")
                time.sleep(0.1)
    
    finally:
        try: 
            client.cancel_all()
        except Exception:
            pass
        client.close()

if __name__ == "__main__":
    run()