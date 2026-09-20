# Bollinger Bands

Bollinger Bands plot a moving average (the mid band, usually a 20-period
SMA) with an upper and lower band placed a fixed number of standard
deviations (usually 2) away from it. The bands widen during volatile periods
and contract during calm ones.

- **%B** expresses where price sits relative to the bands: 0 means price is
  at the lower band, 1 means price is at the upper band, and 0.5 means price
  is at the mid band.
- Prices pressing the upper band for an extended period can indicate a
  strong uptrend rather than an automatic reversal signal, and the same
  applies symmetrically at the lower band.
- A sharp narrowing of the bands ("squeeze") often precedes a volatility
  expansion, though it does not indicate direction.
