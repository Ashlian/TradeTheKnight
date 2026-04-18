"""
Bond Auction Stop-Out Sniper Bot
=================================
Strategy: Bond Auction Stop-Out Sniping (Idea #4)

Overview
--------
Bond auctions on this exchange settle at a *uniform stop-out yield* — every
winning bidder pays the same clearing yield, regardless of what they actually
bid.  The standard auction theory result is that you should bid your true
value; bidding too high (low yield) means you miss the auction entirely, and
bidding exactly at the stop-out gets you filled at the best available coupon.

Most competitor bots either:
  (a) ignore bonds entirely, or
  (b) submit naive yield bids without modelling where the stop-out will land.

This bot exploits that gap by:
  1. Detecting active bond auctions via get_assets().
  2. Estimating the likely stop-out yield from IOR rate + a thin-market
     premium (since participation is sparse, auctions clear at higher yields
     than they would in a deep market).
  3. Bidding just *inside* the expected stop-out — high enough to guarantee
     inclusion, low enough to lock in the best coupon.
  4. Post-issuance: selling the bond on the spot book if/when it trades above
     par after coupon expectations reprice it, or holding for coupon income
     when the carry is favourable vs IOR.

PRD compliance notes
---------------------
- Uses only ExchangeClient() SDK methods (no raw HTTP).
- Reads competition_state from stream_state() and halts during non-live
  states and global halts.
- place_auction_bid(symbol, yield_rate, quantity) is the only auction
  interface; no custom transport.
- Naked short of bonds is NOT used — we only go long (buy at auction or
  secondary market).
- Capital consumption: buy orders lock allocated RUD; we track remaining
  capital before bidding.
- Memory ≤ 256 MB, single file, Python 3.9, approved libs only.

Deployment
----------
1.  Upload this file as your bot on the dashboard.
2.  Set your RUD capital allocation for this bot from the team treasury.
3.  Start the bot from the dashboard (bots launch paused by default).
4.  Tune the constants in the CONFIG block below for your competition.

Environment variables are injected automatically:
    BOT_ID        — used by ExchangeClient() for authentication
    EXCHANGE_URL  — used by ExchangeClient() for the websocket + REST base
"""

import time
import logging
from knight_trader import ExchangeClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("bond_sniper")


# ── CONFIG ────────────────────────────────────────────────────────────────────
# Tune these before competition day.

# How many basis points ABOVE the estimated stop-out to bid.
# Bidding above the stop-out guarantees inclusion; the uniform-price rule
# means you still pay only the stop-out yield, not your bid yield.
# A value of 0.0050 (50 bps) gives comfortable headroom without wasting
# too much potential coupon if your estimate is off.
SNIPE_BUFFER_YIELD = 0.0050  # 50 bps above estimated clearing yield

# Premium added to IOR to estimate clearing yield in a thin market.
# Thin-market auctions tend to clear at IOR + some illiquidity premium.
# Start at 100 bps and narrow as you observe historical stop-outs.
THIN_MARKET_PREMIUM = 0.0100  # 100 bps illiquidity premium over IOR

# Maximum yield we are willing to bid (absolute ceiling).
# Prevents runaway bids if IOR spikes unexpectedly.
MAX_BID_YIELD = 0.20  # 20 % — adjust to match competition instruments

# Fraction of allocated RUD capital to deploy per auction.
# Keeping this below 1.0 preserves capital for other opportunities.
CAPITAL_FRACTION = 0.80

# Bond par value per unit (as stated in PRD: $1000 par per unit).
BOND_PAR = 1000.0

# How often (seconds) to re-scan for new auctions and re-evaluate positions.
SCAN_INTERVAL = 2.0

# Minimum spread between best bid and best ask on the secondary bond market
# before we consider selling into it (avoid crossing a tight spread for nothing).
MIN_SECONDARY_SPREAD_TO_SELL = 0.005  # 0.5 % of par

# How far above par (as a fraction) a bond must trade before we sell.
# Bonds issued below par value (high yield) can re-rate upward.
SELL_PREMIUM_THRESHOLD = 0.01  # sell if price > par * (1 + this)

# ── END CONFIG ────────────────────────────────────────────────────────────────


def get_ior_rate(state: dict) -> float:
    """
    Extract the current IOR rate from stream_state().

    The PRD specifies that ior_rate is the built-in system timeseries and
    that its latest value is included in every stream_state() snapshot.
    """
    timeseries = state.get("timeseries", {})
    ior_entry = timeseries.get("ior_rate")
    if ior_entry is None:
        return 0.0
    # Timeseries values may be a list of (timestamp, value) pairs or a scalar.
    if isinstance(ior_entry, (int, float)):
        return float(ior_entry)
    if isinstance(ior_entry, list) and ior_entry:
        # Take the most recent entry.
        last = ior_entry[-1]
        if isinstance(last, (list, tuple)):
            return float(last[-1])
        return float(last)
    return 0.0


def estimate_stop_out_yield(ior_rate: float) -> float:
    """
    Estimate where the auction stop-out yield will land.

    Model: stop_out ≈ IOR + thin_market_premium
    This reflects that rational bidders demand at least IOR (the risk-free
    alternative) plus a premium for illiquidity and credit uncertainty.
    Capped at MAX_BID_YIELD.
    """
    estimated = ior_rate + THIN_MARKET_PREMIUM
    return min(estimated, MAX_BID_YIELD)


def compute_bid_yield(estimated_stop_out: float) -> float:
    """
    Our actual bid yield = estimated stop-out + snipe buffer.

    The buffer guarantees we land above the clearing yield so we get filled.
    Under the uniform-price rule we still receive the stop-out coupon rate,
    not the yield we bid, so the buffer costs us nothing on the coupon —
    it only helps ensure inclusion.
    """
    bid = estimated_stop_out + SNIPE_BUFFER_YIELD
    return min(bid, MAX_BID_YIELD)


def compute_bid_quantity(team_state: dict) -> float:
    """
    Compute how many bond units to bid for, based on available capital.

    PRD: buy orders lock the bot's allocated RUD capital.
    We use CAPITAL_FRACTION of whatever is currently free.
    Units = floor(free_capital * fraction / par), snapped to 0.001.
    """
    capital = float(team_state.get("capital", 0.0))
    locked = float(team_state.get("locked_capital", 0.0))
    free = max(capital - locked, 0.0)
    raw_units = (free * CAPITAL_FRACTION) / BOND_PAR
    # Snap to 0.001 as required by PRD.
    snapped = round(raw_units - (raw_units % 0.001), 3)
    return max(snapped, 0.0)


def find_bond_auctions(assets: list) -> list:
    """
    Identify assets that are bond auctions currently open for bidding.

    The PRD describes bonds as admin-created issues auctioned at $1000 par.
    We look for asset type 'bond' (or 'BOND') with an active auction flag.
    Adjust the field names to match whatever the SDK returns for your exchange.
    """
    auctions = []
    for asset in assets:
        asset_type = str(asset.get("type", "")).lower()
        if asset_type != "bond":
            continue
        # The exchange sets an 'auction_open' or similar flag when bidding is live.
        # Fall back to checking for a 'when_issued' or 'auction' status field.
        status = str(asset.get("status", "")).lower()
        auction_open = asset.get("auction_open", False)
        if auction_open or status in ("auction", "when_issued", "bidding"):
            auctions.append(asset)
    return auctions


def find_secondary_bonds(assets: list, state: dict) -> list:
    """
    Identify bonds that have already been issued and are trading on the
    secondary spot-style order book.
    """
    secondary = []
    for asset in assets:
        asset_type = str(asset.get("type", "")).lower()
        if asset_type != "bond":
            continue
        status = str(asset.get("status", "")).lower()
        # Issued bonds trade on a normal order book.
        if status in ("issued", "live", "trading", "active"):
            symbol = asset.get("symbol") or asset.get("id")
            if symbol:
                secondary.append(symbol)
    return secondary


class BondAuctionSniper:
    def __init__(self):
        self.client = ExchangeClient()
        # Track which auctions we have already placed bids in this session.
        self.bids_placed: set = set()
        log.info("Bond Auction Sniper initialised.")

    def run(self):
        next_scan = 0.0

        for state in self.client.stream_state():
            try:
                comp_state = state.get("competition_state", "")
                if comp_state != "live":
                    log.debug("Competition not live (%s) — waiting.", comp_state)
                    time.sleep(0.5)
                    continue

                now = time.monotonic()
                if now < next_scan:
                    continue
                next_scan = now + SCAN_INTERVAL

                ior_rate = get_ior_rate(state)
                log.info("IOR rate: %.4f (%.2f%%)", ior_rate, ior_rate * 100)

                assets = self.client.get_assets()
                team_state = self.client.get_team_state()

                # ── 1. Participate in open auctions ──────────────────────────
                auctions = find_bond_auctions(assets)
                for asset in auctions:
                    symbol = asset.get("symbol") or asset.get("id")
                    if not symbol:
                        continue
                    if symbol in self.bids_placed:
                        log.debug("Already bid on %s this session.", symbol)
                        continue

                    est_stop_out = estimate_stop_out_yield(ior_rate)
                    bid_yield = compute_bid_yield(est_stop_out)
                    quantity = compute_bid_quantity(team_state)

                    if quantity < 0.001:
                        log.warning(
                            "Insufficient capital to bid on %s — skipping.", symbol
                        )
                        continue

                    log.info(
                        "AUCTION %s | est_stop_out=%.4f  bid_yield=%.4f  qty=%.3f",
                        symbol,
                        est_stop_out,
                        bid_yield,
                        quantity,
                    )
                    try:
                        self.client.place_auction_bid(symbol, bid_yield, quantity)
                        self.bids_placed.add(symbol)
                        log.info("Bid placed on %s.", symbol)
                    except Exception as exc:
                        log.error("place_auction_bid(%s) failed: %s", symbol, exc)

                # ── 2. Secondary-market exit logic ────────────────────────────
                # After the auction clears and the bond is issued, look for
                # opportunities to sell inventory above par if the market has
                # repriced upward (yield fell → price rose).
                secondary_symbols = find_secondary_bonds(assets, state)
                for symbol in secondary_symbols:
                    self._consider_secondary_exit(state, symbol)

            except Exception as exc:
                log.error("Main loop error: %s", exc)
                time.sleep(1.0)

    def _consider_secondary_exit(self, state: dict, symbol: str):
        """
        If we hold inventory in a secondary bond and its market price has
        risen above par by SELL_PREMIUM_THRESHOLD, post a limit sell.

        PRD: sell() takes (symbol, price, quantity).
        We place at best_bid to guarantee a fill rather than waiting at ask.
        """
        # Check our inventory for this symbol.
        inventory = state.get("inventory", {})
        position = float(inventory.get(symbol, 0.0))
        if position <= 0:
            return  # No long position to exit.

        best_bid = self.client.get_best_bid(symbol)
        if best_bid is None:
            return

        best_bid = float(best_bid)
        sell_threshold = BOND_PAR * (1.0 + SELL_PREMIUM_THRESHOLD)

        if best_bid < sell_threshold:
            return  # Not rich enough yet.

        best_ask = self.client.get_best_ask(symbol)
        if best_ask is None:
            return
        best_ask = float(best_ask)

        spread = best_ask - best_bid
        if spread < MIN_SECONDARY_SPREAD_TO_SELL * BOND_PAR:
            return  # Spread too tight — not worth crossing.

        # Sell our full position at best_bid (aggressive limit = immediate fill).
        sell_qty = round(position - (position % 0.001), 3)
        if sell_qty < 0.001:
            return

        log.info(
            "SECONDARY EXIT %s | best_bid=%.4f  threshold=%.4f  qty=%.3f",
            symbol,
            best_bid,
            sell_threshold,
            sell_qty,
        )
        try:
            self.client.sell(symbol, best_bid, sell_qty)
        except Exception as exc:
            log.error("sell(%s) failed: %s", symbol, exc)


def run():
    bot = BondAuctionSniper()
    bot.run()


if __name__ == "__main__":
    run()
