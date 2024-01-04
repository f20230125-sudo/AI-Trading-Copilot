# Moving Averages (SMA and EMA)

A **Simple Moving Average (SMA)** is the unweighted mean of price over a
fixed lookback window (e.g., 20 or 50 days). It smooths short-term noise to
reveal the underlying trend direction.

An **Exponential Moving Average (EMA)** weights recent prices more heavily,
so it reacts faster to new information than an SMA of the same length.

- Price trading above its 50-day SMA is commonly read as a sign of an
  established uptrend; trading below it, a downtrend.
- Shorter moving averages crossing above longer ones (a "golden cross") or
  below them (a "death cross") are widely watched trend-change signals,
  though they lag actual turning points because they are averages of past
  price.
- `price_vs_sma20` in this project expresses the current price's percentage
  distance from its 20-day average, a normalized measure of short-term
  trend strength.
