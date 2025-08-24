export async function getMidPriceGate({ base, quote }) {
    console.log(base, quote, "SOME");
    const url = `https://api.gateio.ws/api/v4/spot/order_book?currency_pair=${base}_${quote}&limit=50`;
    const res = await fetch(url);
    if (!res.ok)
        return null;
    const data = await res.json();
    // Gate order_book returns { bids: [[price, size], ...], asks: [[price, size], ...], ... }
    if (!data.bids?.length || !data.asks?.length)
        return null;
    const bid = parseFloat(data.bids[0][0]);
    const ask = parseFloat(data.asks[0][0]);
    return { bid, ask, mid: (bid + ask) / 2 };
}
