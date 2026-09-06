NUMERIC_FEATURES = [
    "elevation_m",
    "slope_degrees",
    "aspect_degrees",
    "rainfall_1d_before",
    "rainfall_3d_before",
    "rainfall_7d_before",
    "rainfall_14d_before",
    "rainfall_30d_before",
    "rainfall_7d_max1d",
    "rainfall_3d_over_7d_ratio",
]

ALL_FEATURES = NUMERIC_FEATURES

TIER_ORDER = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}