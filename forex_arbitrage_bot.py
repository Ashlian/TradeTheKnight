import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from knight_trader import ExchangeClient


SYMBOLS: Tuple[str, ...] = ("COORUD", "COLRUD", "DOURUD")
PAIRS: Tuple[Tuple[str, str], ...] = (
    ("COORUD", "COLRUD"),
    ("COORUD", "DOURUD"),
    ("COLRUD", "DOURUD"),
)

TICK_SIZE = 0.0001
REFRESH_SECS = float(os.environ.get("FX_ARB_REFRESH_SECS", "0.35"))
MODEL_ALPHA = float(os.environ.get("FX_ARB_MODEL_ALPHA", "0.04"))
ENTRY_Z = float(os.environ.get("FX_ARB_ENTRY_Z", "2.2"))
EXIT_Z = float(os.environ.get("FX_ARB_EXIT_Z", "0.7"))
MIN_OBSERVATIONS = int(os.environ.get("FX_ARB_MIN_OBS", "30"))
MAX_POSITION = float(os.environ.get("FX_ARB_MAX_POSITION", "4.0"))
MAX_ORDER_QTY = float(os.environ.get("FX_ARB_MAX_ORDER_QTY", "0.75"))
MAX_GROSS_NOTIONAL = float(os.environ.get("FX_ARB_MAX_GROSS_NOTIONAL", "20.0"))
TOP_QTY_FRACTION = float(os.environ.get("FX_ARB_TOP_QTY_FRACTION", "0.6"))
MIN_TRADE_QTY = float(os.environ.get("FX_ARB_MIN_TRADE_QTY", "0.05"))


@dataclass
class PairModel:
    mean: float = 0.0
    variance: float = 1e-6
    observations: int = 0

    def update(self, value: float) -> None:
        if self.observations == 0:
            self.mean = value
            self.variance = 1e-6
            self.observations = 1
            return

        previous_mean = self.mean
        self.mean = (1.0 - MODEL_ALPHA) * self.mean + MODEL_ALPHA * value
        delta = value - previous_mean
        self.variance = max(1e-8, (1.0 - MODEL_ALPHA) * self.variance + MODEL_ALPHA * delta * delta)
        self.observations += 1

    def zscore(self, value: float) -> float:
        if self.observations < MIN_OBSERVATIONS:
            return 0.0
        stddev = math.sqrt(max(self.variance, 1e-8))
        return (value - self.mean) / stddev


class ForexArbitrageBot:
    def __init__(self, client: ExchangeClient):
        self.client = client
        self.bot_id = str(os.environ.get("BOT_ID") or "")
        self.models: Dict[Tuple[str, str], PairModel] = {pair: PairModel() for pair in PAIRS}
        self.positions: Dict[str, float] = {symbol: 0.0 for symbol in SYMBOLS}
        self.open_orders: Dict[str, Set[str]] = {symbol: set() for symbol in SYMBOLS}
        self.seen_trade_keys: Set[Tuple[object, ...]] = set()
        self.trade_key_queue: deque[Tuple[object, ...]] = deque(maxlen=5000)
        self.next_refresh = 0.0

    def run(self) -> None:
        for state in self.client.stream_state():
            try:
                self.on_state(state)
            except Exception as exc:
                print(f"bot error: {exc}")
                time.sleep(1.0)

    def on_state(self, state: Dict[str, object]) -> None:
        self._ingest_trades(state)

        competition_state = str(state.get("competition_state", "pre_open"))
        if competition_state != "live":
            self._cancel_stale_orders()
            return

        now = time.monotonic()
        if now < self.next_refresh:
            return

        books = state.get("book", {})
        markets = self._read_markets(books if isinstance(books, dict) else {})
        if len(markets) != len(SYMBOLS):
            return

        self._update_models(markets)

        self._cancel_stale_orders()

        best_pair = self._best_signal(markets)
        gross_notional = self._gross_notional(markets)

        if best_pair is None:
            if gross_notional > 0.0:
                self._flatten_positions(markets)
            self.next_refresh = now + REFRESH_SECS
            return

        pair, zscore = best_pair
        if gross_notional > MAX_GROSS_NOTIONAL or abs(zscore) < EXIT_Z:
            self._flatten_positions(markets)
            self.next_refresh = now + REFRESH_SECS
            return

        left, right = pair
        if zscore > 0.0:
            rich_symbol, cheap_symbol = left, right
        else:
            rich_symbol, cheap_symbol = right, left

        rich_market = markets[rich_symbol]
        cheap_market = markets[cheap_symbol]
        trade_qty = self._trade_size(rich_symbol, cheap_symbol, rich_market, cheap_market, gross_notional)

        if trade_qty < MIN_TRADE_QTY:
            self._flatten_positions(markets)
            self.next_refresh = now + REFRESH_SECS
            return

        if rich_market["best_bid"] > 0.0:
            order_id = self.client.sell(rich_symbol, rich_market["best_bid"], trade_qty)
            if order_id:
                self.open_orders[rich_symbol].add(order_id)

        if cheap_market["best_ask"] > 0.0:
            order_id = self.client.buy(cheap_symbol, cheap_market["best_ask"], trade_qty)
            if order_id:
                self.open_orders[cheap_symbol].add(order_id)

        self.next_refresh = now + REFRESH_SECS

    def _read_markets(self, books: Dict[str, object]) -> Dict[str, Dict[str, float]]:
        markets: Dict[str, Dict[str, float]] = {}
        for symbol in SYMBOLS:
            book = books.get(symbol)
            if not isinstance(book, dict):
                continue

            best_bid, bid_qty = self._best_level(book.get("bids", {}), reverse=True)
            best_ask, ask_qty = self._best_level(book.get("asks", {}), reverse=False)
            if best_bid <= 0.0 or best_ask <= 0.0 or best_ask <= best_bid:
                continue

            markets[symbol] = {
                "best_bid": best_bid,
                "best_ask": best_ask,
                "bid_qty": bid_qty,
                "ask_qty": ask_qty,
                "mid": (best_bid + best_ask) / 2.0,
                "spread": best_ask - best_bid,
            }
        return markets

    def _best_level(self, side: object, reverse: bool) -> Tuple[float, float]:
        if not isinstance(side, dict) or not side:
            return 0.0, 0.0

        ranked_levels: List[Tuple[float, object]] = sorted(
            ((float(price), orders) for price, orders in side.items()),
            key=lambda item: item[0],
            reverse=reverse,
        )
        if not ranked_levels:
            return 0.0, 0.0

        price, raw_orders = ranked_levels[0]
        qty = 0.0
        if isinstance(raw_orders, list):
            for order in raw_orders:
                if isinstance(order, dict):
                    qty += float(order.get("quantity", 0.0) or 0.0)
        return price, qty

    def _update_models(self, markets: Dict[str, Dict[str, float]]) -> None:
        for left, right in PAIRS:
            spread = math.log(markets[left]["mid"]) - math.log(markets[right]["mid"])
            self.models[(left, right)].update(spread)

    def _best_signal(self, markets: Dict[str, Dict[str, float]]) -> Optional[Tuple[Tuple[str, str], float]]:
        best: Optional[Tuple[Tuple[str, str], float]] = None
        best_abs = 0.0

        for left, right in PAIRS:
            spread = math.log(markets[left]["mid"]) - math.log(markets[right]["mid"])
            zscore = self.models[(left, right)].zscore(spread)
            if abs(zscore) <= best_abs:
                continue

            execution_cost = (
                markets[left]["spread"] / markets[left]["mid"]
                + markets[right]["spread"] / markets[right]["mid"]
            )
            cost_buffer = max(ENTRY_Z, 1.0 + 8.0 * execution_cost)
            if abs(zscore) < cost_buffer:
                continue

            best = ((left, right), zscore)
            best_abs = abs(zscore)

        return best

    def _trade_size(
        self,
        rich_symbol: str,
        cheap_symbol: str,
        rich_market: Dict[str, float],
        cheap_market: Dict[str, float],
        gross_notional: float,
    ) -> float:
        rich_room = MAX_POSITION + self.positions[rich_symbol]
        cheap_room = MAX_POSITION - self.positions[cheap_symbol]
        room_qty = min(rich_room, cheap_room)

        visible_qty = min(
            rich_market["bid_qty"] * TOP_QTY_FRACTION,
            cheap_market["ask_qty"] * TOP_QTY_FRACTION,
        )

        notional_room = max(0.0, MAX_GROSS_NOTIONAL - gross_notional)
        pair_notional = max(rich_market["mid"] + cheap_market["mid"], 1e-6)
        notional_qty = notional_room / pair_notional if notional_room > 0.0 else 0.0

        qty = min(MAX_ORDER_QTY, room_qty, visible_qty, notional_qty)
        return self._snap_quantity(qty)

    def _flatten_positions(self, markets: Dict[str, Dict[str, float]]) -> None:
        for symbol in SYMBOLS:
            position = self.positions[symbol]
            if abs(position) < MIN_TRADE_QTY:
                continue

            market = markets.get(symbol)
            if not market:
                continue

            if position > 0.0 and market["best_bid"] > 0.0:
                qty = self._snap_quantity(min(position, MAX_ORDER_QTY, market["bid_qty"] * TOP_QTY_FRACTION))
                if qty >= MIN_TRADE_QTY:
                    order_id = self.client.sell(symbol, market["best_bid"], qty)
                    if order_id:
                        self.open_orders[symbol].add(order_id)
            elif position < 0.0 and market["best_ask"] > 0.0:
                qty = self._snap_quantity(min(-position, MAX_ORDER_QTY, market["ask_qty"] * TOP_QTY_FRACTION))
                if qty >= MIN_TRADE_QTY:
                    order_id = self.client.buy(symbol, market["best_ask"], qty)
                    if order_id:
                        self.open_orders[symbol].add(order_id)

    def _gross_notional(self, markets: Dict[str, Dict[str, float]]) -> float:
        total = 0.0
        for symbol in SYMBOLS:
            market = markets.get(symbol)
            if not market:
                continue
            total += abs(self.positions[symbol]) * market["mid"]
        return total

    def _ingest_trades(self, state: Dict[str, object]) -> None:
        if not self.bot_id:
            return

        trades = state.get("trades", [])
        if not isinstance(trades, list):
            return

        for trade in reversed(trades):
            if not isinstance(trade, dict):
                continue

            symbol = str(trade.get("symbol", "")).upper()
            if symbol not in self.positions:
                continue

            trade_key = (
                trade.get("tick"),
                trade.get("timestamp"),
                symbol,
                trade.get("buyer_id"),
                trade.get("seller_id"),
                trade.get("price"),
                trade.get("quantity"),
            )
            if trade_key in self.seen_trade_keys:
                continue

            if len(self.trade_key_queue) == self.trade_key_queue.maxlen:
                expired_key = self.trade_key_queue.popleft()
                self.seen_trade_keys.discard(expired_key)

            self.trade_key_queue.append(trade_key)
            self.seen_trade_keys.add(trade_key)

            quantity = float(trade.get("quantity", 0.0) or 0.0)
            if str(trade.get("buyer_id", "")) == self.bot_id:
                self.positions[symbol] += quantity
            if str(trade.get("seller_id", "")) == self.bot_id:
                self.positions[symbol] -= quantity

    def _cancel_stale_orders(self) -> None:
        for symbol in SYMBOLS:
            active_ids = list(self.open_orders[symbol])
            self.open_orders[symbol].clear()
            for order_id in active_ids:
                self.client.cancel(order_id)

    @staticmethod
    def _snap_quantity(quantity: float) -> float:
        if quantity <= 0.0:
            return 0.0
        return math.floor(quantity * 1000.0) / 1000.0


def run() -> None:
    client = ExchangeClient()
    bot = ForexArbitrageBot(client)
    try:
        bot.run()
    finally:
        bot._cancel_stale_orders()
        client.close()


if __name__ == "__main__":
    run()
