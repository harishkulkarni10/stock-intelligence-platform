from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32

ticker = Entity(name="ticker", join_keys=["ticker"], value_type=ValueType.STRING)

stock_stats_source = FileSource(
    name="stock_stats_source",
    path="data/features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

stock_stats_fv = FeatureView(
    name="stock_stats",
    entities=[ticker],
    ttl=timedelta(days=365),
    schema=[
        Field(name="Open", dtype=Float32),
        Field(name="High", dtype=Float32),
        Field(name="Low", dtype=Float32),
        Field(name="Close", dtype=Float32),
        Field(name="Volume", dtype=Float32),
        Field(name="RSI14", dtype=Float32),
        Field(name="MACD", dtype=Float32),
    ],
    online=True,
    source=stock_stats_source,
    tags={"team": "mlops"},
)
