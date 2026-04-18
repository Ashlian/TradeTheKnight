import json
import struct
from typing import Any, Dict, List, Optional, Tuple


PRICE_SCALE = 10_000
QUANTITY_SCALE = 1_000
COMPETITION_STATES = {
    0: "pre_open",
    1: "live",
    2: "paused",
    3: "post_close",
}
ORDER_STATUSES = {
    0: "accepted",
    1: "rejected",
    2: "canceled",
    3: "filled",
}


def decode_state_message(message: Any) -> Dict[str, Any]:
    if isinstance(message, str):
        return json.loads(message)
    if isinstance(message, (bytes, bytearray)):
        raw = bytes(message)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            v2 = decode_state_v2_envelope(raw)
            if v2 is not None:
                return v2
            return decode_unified_state(raw)
    raise TypeError(f"Unsupported websocket payload type: {type(message)!r}")


def decode_state_v2_envelope(data: bytes) -> Optional[Dict[str, Any]]:
    envelope: Dict[str, Any] = {
        "type": None,
        "seq": 0,
        "tick": 0,
        "ts_ms": 0,
        "competition_state": "live",
        "payload": {},
    }
    pos = 0
    saw_payload = False

    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field = tag >> 3
        wire = tag & 0x07

        if field == 1 and wire == 0:
            envelope["seq"], pos = _read_varint(data, pos)
        elif field == 2 and wire == 0:
            envelope["tick"], pos = _read_varint(data, pos)
        elif field == 3 and wire == 0:
            envelope["ts_ms"], pos = _read_varint(data, pos)
        elif field == 4 and wire == 0:
            state, pos = _read_varint(data, pos)
            envelope["competition_state"] = COMPETITION_STATES.get(state, "live")
        elif field == 10 and wire == 2:
            payload, pos = _read_length_delimited(data, pos)
            envelope["type"] = "snapshot"
            envelope["payload"] = _decode_state_v2_snapshot(payload)
            saw_payload = True
        elif field == 11 and wire == 2:
            payload, pos = _read_length_delimited(data, pos)
            envelope["type"] = "delta"
            envelope["payload"] = _decode_state_v2_delta(payload)
            saw_payload = True
        elif field == 12 and wire == 2:
            _, pos = _read_length_delimited(data, pos)
            envelope["type"] = "heartbeat"
            envelope["payload"] = {}
            saw_payload = True
        elif field == 13 and wire == 2:
            payload, pos = _read_length_delimited(data, pos)
            envelope["type"] = "resync_required"
            envelope["payload"] = _decode_state_v2_resync(payload)
            saw_payload = True
        else:
            pos = _skip_value(data, pos, wire)

    return envelope if saw_payload else None


def _decode_state_v2_snapshot(data: bytes) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "books": {},
        "timeseries": {},
        "meta": {"price_scale": PRICE_SCALE, "quantity_scale": QUANTITY_SCALE},
        "tape": [],
    }
    pos = 0

    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field = tag >> 3
        wire = tag & 0x07

        if field == 1 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            symbol, book = _decode_state_v2_book_state(raw)
            payload["books"][symbol] = book
        elif field == 2 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            key, value = _decode_state_v2_timeseries_value(raw)
            payload["timeseries"][key] = value
        elif field == 3 and wire == 0:
            payload["meta"]["price_scale"], pos = _read_varint(data, pos)
        elif field == 4 and wire == 0:
            payload["meta"]["quantity_scale"], pos = _read_varint(data, pos)
        elif field == 5 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            payload["tape"].append(_decode_state_v2_tape_record(raw))
        else:
            pos = _skip_value(data, pos, wire)

    return payload


def _decode_state_v2_delta(data: bytes) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "book_updates": [],
        "timeseries_updates": [],
        "tape": [],
        "order_updates": [],
    }
    pos = 0

    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field = tag >> 3
        wire = tag & 0x07

        if field == 1 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            payload["book_updates"].append(_decode_state_v2_book_delta(raw))
        elif field == 2 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            key, value = _decode_state_v2_timeseries_value(raw)
            payload["timeseries_updates"].append({"key": key, "value": value})
        elif field == 3 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            payload["tape"].append(_decode_state_v2_tape_record(raw))
        elif field == 4 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            payload["order_updates"].append(_decode_state_v2_order_status_update(raw))
        else:
            pos = _skip_value(data, pos, wire)

    return payload


def _decode_state_v2_resync(data: bytes) -> Dict[str, Any]:
    reason = ""
    pos = 0
    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field = tag >> 3
        wire = tag & 0x07
        if field == 1 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            reason = raw.decode("utf-8")
        else:
            pos = _skip_value(data, pos, wire)
    return {"reason": reason}


def _decode_state_v2_book_state(data: bytes) -> Tuple[str, Dict[str, Any]]:
    symbol = ""
    bids: Dict[int, int] = {}
    asks: Dict[int, int] = {}
    last_trade_ticks = None
    pos = 0

    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field = tag >> 3
        wire = tag & 0x07

        if field == 1 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            symbol = raw.decode("utf-8")
        elif field == 2 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            price_ticks, quantity_lots = _decode_state_v2_book_level(raw)
            bids[price_ticks] = quantity_lots
        elif field == 3 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            price_ticks, quantity_lots = _decode_state_v2_book_level(raw)
            asks[price_ticks] = quantity_lots
        elif field == 4 and wire == 0:
            last_trade_ticks, pos = _read_varint(data, pos)
        else:
            pos = _skip_value(data, pos, wire)

    return symbol, {
        "bids": bids,
        "asks": asks,
        "last_trade_ticks": last_trade_ticks,
    }


def _decode_state_v2_book_delta(data: bytes) -> Dict[str, Any]:
    symbol = ""
    bids: List[List[int]] = []
    asks: List[List[int]] = []
    last_trade_ticks = None
    pos = 0

    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field = tag >> 3
        wire = tag & 0x07

        if field == 1 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            symbol = raw.decode("utf-8")
        elif field == 2 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            bids.append(list(_decode_state_v2_book_level(raw)))
        elif field == 3 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            asks.append(list(_decode_state_v2_book_level(raw)))
        elif field == 4 and wire == 0:
            last_trade_ticks, pos = _read_varint(data, pos)
        else:
            pos = _skip_value(data, pos, wire)

    return {
        "symbol": symbol,
        "bids": bids,
        "asks": asks,
        "last_trade_ticks": last_trade_ticks,
    }


def _decode_state_v2_book_level(data: bytes) -> Tuple[int, int]:
    price_ticks = 0
    quantity_lots = 0
    pos = 0
    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field = tag >> 3
        wire = tag & 0x07
        if field == 1 and wire == 0:
            price_ticks, pos = _read_varint(data, pos)
        elif field == 2 and wire == 0:
            quantity_lots, pos = _read_varint(data, pos)
        else:
            pos = _skip_value(data, pos, wire)
    return price_ticks, quantity_lots


def _decode_state_v2_timeseries_value(data: bytes) -> Tuple[str, float]:
    key = ""
    value = 0
    pos = 0
    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field = tag >> 3
        wire = tag & 0x07
        if field == 1 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            key = raw.decode("utf-8")
        elif field == 2 and wire == 0:
            value, pos = _read_varint(data, pos)
        else:
            pos = _skip_value(data, pos, wire)
    return key, value / PRICE_SCALE


def _decode_state_v2_tape_record(data: bytes) -> Dict[str, Any]:
    record = {
        "symbol": "",
        "side": "buy",
        "price_ticks": 0,
        "quantity_lots": 0,
        "buyer_bot_id": 0,
        "seller_bot_id": 0,
        "tick": 0,
        "ts_ms": 0,
    }
    pos = 0
    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field = tag >> 3
        wire = tag & 0x07

        if field == 1 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            record["symbol"] = raw.decode("utf-8")
        elif field == 2 and wire == 0:
            side, pos = _read_varint(data, pos)
            record["side"] = "buy" if side == 0 else "sell"
        elif field == 3 and wire == 0:
            record["price_ticks"], pos = _read_varint(data, pos)
        elif field == 4 and wire == 0:
            record["quantity_lots"], pos = _read_varint(data, pos)
        elif field == 5 and wire == 0:
            record["buyer_bot_id"], pos = _read_varint(data, pos)
        elif field == 6 and wire == 0:
            record["seller_bot_id"], pos = _read_varint(data, pos)
        elif field == 7 and wire == 0:
            record["tick"], pos = _read_varint(data, pos)
        elif field == 8 and wire == 0:
            record["ts_ms"], pos = _read_varint(data, pos)
        else:
            pos = _skip_value(data, pos, wire)
    return record


def _decode_state_v2_order_status_update(data: bytes) -> Dict[str, Any]:
    update = {
        "order_id": "",
        "client_order_id": "",
        "symbol": "",
        "side": "buy",
        "status": "accepted",
        "reason": None,
        "price_ticks": None,
        "quantity_lots": None,
    }
    pos = 0
    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field = tag >> 3
        wire = tag & 0x07
        if field == 1 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            update["order_id"] = raw.decode("utf-8")
        elif field == 2 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            update["client_order_id"] = raw.decode("utf-8")
        elif field == 3 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            update["symbol"] = raw.decode("utf-8")
        elif field == 4 and wire == 0:
            side, pos = _read_varint(data, pos)
            update["side"] = "buy" if side == 0 else "sell"
        elif field == 5 and wire == 0:
            status, pos = _read_varint(data, pos)
            update["status"] = ORDER_STATUSES.get(status, "accepted")
        elif field == 6 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            update["reason"] = raw.decode("utf-8")
        elif field == 7 and wire == 0:
            update["price_ticks"], pos = _read_varint(data, pos)
        elif field == 8 and wire == 0:
            update["quantity_lots"], pos = _read_varint(data, pos)
        else:
            pos = _skip_value(data, pos, wire)
    return update


def decode_unified_state(data: bytes) -> Dict[str, Any]:
    state: Dict[str, Any] = {"tick": 0, "book": {}, "timeseries": {}, "trades": []}
    pos = 0

    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field = tag >> 3
        wire = tag & 0x07

        if field == 1 and wire == 0:
            state["tick"], pos = _read_varint(data, pos)
        elif field == 2 and wire == 2:
            payload, pos = _read_length_delimited(data, pos)
            symbol, book = _decode_book_snapshot(payload)
            state["book"][symbol] = book
        elif field == 3 and wire == 2:
            payload, pos = _read_length_delimited(data, pos)
            name, value = _decode_timeseries_point(payload)
            state["timeseries"][name] = value
        elif field == 4 and wire == 2:
            payload, pos = _read_length_delimited(data, pos)
            state["trades"].append(_decode_trade(payload))
        else:
            pos = _skip_value(data, pos, wire)

    return state


def _decode_book_snapshot(data: bytes) -> Tuple[str, Dict[str, Any]]:
    symbol = ""
    bids: Dict[str, List[Dict[str, Any]]] = {}
    asks: Dict[str, List[Dict[str, Any]]] = {}
    pos = 0

    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field = tag >> 3
        wire = tag & 0x07

        if field == 1 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            symbol = raw.decode("utf-8")
        elif field in (2, 3) and wire == 2:
            payload, pos = _read_length_delimited(data, pos)
            price, orders = _decode_price_level(payload, symbol, "Bid" if field == 2 else "Ask")
            if field == 2:
                bids[price] = orders
            else:
                asks[price] = orders
        else:
            pos = _skip_value(data, pos, wire)

    return symbol, {"bids": bids, "asks": asks}


def _decode_price_level(data: bytes, symbol: str, side: str) -> Tuple[str, List[Dict[str, Any]]]:
    price = 0.0
    orders: List[Dict[str, Any]] = []
    pos = 0

    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field = tag >> 3
        wire = tag & 0x07

        if field == 1 and wire == 1:
            price, pos = _read_double(data, pos)
        elif field == 2 and wire == 2:
            payload, pos = _read_length_delimited(data, pos)
            orders.append(_decode_order(payload, symbol, side, price))
        else:
            pos = _skip_value(data, pos, wire)

    return _format_number(price), orders


def _decode_order(data: bytes, symbol: str, side: str, price: float) -> Dict[str, Any]:
    order = {
        "id": "",
        "owner_id": "",
        "symbol": symbol,
        "price": _format_number(price),
        "quantity": "0",
        "side": side,
        "timestamp": 0,
    }
    pos = 0

    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field = tag >> 3
        wire = tag & 0x07

        if field == 1 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            order["id"] = raw.decode("utf-8")
        elif field == 2 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            order["owner_id"] = raw.decode("utf-8")
        elif field == 3 and wire == 1:
            quantity, pos = _read_double(data, pos)
            order["quantity"] = _format_number(quantity)
        elif field == 4 and wire == 0:
            timestamp, pos = _read_varint(data, pos)
            order["timestamp"] = timestamp
        else:
            pos = _skip_value(data, pos, wire)

    return order


def _decode_timeseries_point(data: bytes) -> Tuple[str, float]:
    name = ""
    value = 0.0
    pos = 0

    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field = tag >> 3
        wire = tag & 0x07

        if field == 1 and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            name = raw.decode("utf-8")
        elif field == 2 and wire == 1:
            value, pos = _read_double(data, pos)
        else:
            pos = _skip_value(data, pos, wire)

    return name, value


def _decode_trade(data: bytes) -> Dict[str, Any]:
    trade = {
        "bid_order_id": "",
        "ask_order_id": "",
        "buyer_id": "",
        "seller_id": "",
        "symbol": "",
        "price": "0",
        "quantity": "0",
        "timestamp": 0,
        "tick": 0,
    }
    pos = 0

    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field = tag >> 3
        wire = tag & 0x07

        if field in (1, 2, 3, 4, 5) and wire == 2:
            raw, pos = _read_length_delimited(data, pos)
            value = raw.decode("utf-8")
            if field == 1:
                trade["bid_order_id"] = value
            elif field == 2:
                trade["ask_order_id"] = value
            elif field == 3:
                trade["buyer_id"] = value
            elif field == 4:
                trade["seller_id"] = value
            else:
                trade["symbol"] = value
        elif field == 6 and wire == 1:
            price, pos = _read_double(data, pos)
            trade["price"] = _format_number(price)
        elif field == 7 and wire == 1:
            quantity, pos = _read_double(data, pos)
            trade["quantity"] = _format_number(quantity)
        elif field == 8 and wire == 0:
            trade["timestamp"], pos = _read_varint(data, pos)
        elif field == 9 and wire == 0:
            trade["tick"], pos = _read_varint(data, pos)
        else:
            pos = _skip_value(data, pos, wire)

    return trade


def _read_length_delimited(data: bytes, pos: int) -> Tuple[bytes, int]:
    length, pos = _read_varint(data, pos)
    end = pos + length
    return data[pos:end], end


def _read_double(data: bytes, pos: int) -> Tuple[float, int]:
    return struct.unpack("<d", data[pos:pos + 8])[0], pos + 8


def _read_varint(data: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0

    while True:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return result, pos
        shift += 7


def _skip_value(data: bytes, pos: int, wire: int) -> int:
    if wire == 0:
        _, pos = _read_varint(data, pos)
        return pos
    if wire == 1:
        return pos + 8
    if wire == 2:
        length, pos = _read_varint(data, pos)
        return pos + length
    if wire == 5:
        return pos + 4
    raise ValueError(f"Unsupported protobuf wire type: {wire}")


def _format_number(value: float) -> str:
    return format(value, ".15g")