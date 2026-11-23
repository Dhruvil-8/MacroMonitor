# Learning Note: Seasonal Inventory Bands in Commodity Markets

## The Economic Logic of Seasonal Envelopes
Physical commodities (crude oil, refined fuels, agricultural grains, and natural gas) follow pronounced annual seasonal consumption and production cycles:
- **Natural Gas**: Builds during spring/summer injection season and draws rapidly during winter heating demand.
- **Gasoline**: Builds in late winter/early spring ahead of peak summer driving season.
- **Crude Oil**: Swings with refinery turnaround and maintenance periods (spring and autumn).

## Calculating 5-Year Bands Without Contamination
Comparing inventory levels to the raw previous week is misleading because storage is naturally seasonal. A true benchmark requires comparing current storage in week $w$ to the distribution of storage in the exact same week $w$ across previous years.

$$\text{Band}(w) = \left[ \min_{y \in [Y-5, Y-1]} S_{y,w}, \; \max_{y \in [Y-5, Y-1]} S_{y,w} \right]$$
$$\text{Mean}(w) = \frac{1}{5} \sum_{y=Y-5}^{Y-1} S_{y,w}$$

Crucially, the current year $Y$ must be strictly omitted from the band calculations. Doing so prevents in-progress physical supply crunches or gluts from distorting the historical baseline.
