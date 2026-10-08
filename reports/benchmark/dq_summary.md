# Data-quality summary - METR-LA benchmark

Period: 2012-03-01 00:00:00 to 2012-06-27 23:55:00 (5-min steps, naive timestamps treated as America/Los_Angeles local time).

## Rows per source

| Source | Rows | Notes |
|---|---|---|
| METR-LA speeds (all sensors) | 34,272 timestamps x 207 sensors = 7,094,304 readings | mph |
| METR-LA speeds (used) | 34,272 x 40 = 1,370,880 readings | random_state=42 |
| Open-Meteo weather | 2,856 hourly values (as joined) | archive API, LA 34.05,-118.25 |
| US holidays 2012 | 12 dates | `holidays` lib; 11,520 sensor-rows fall on one |

## Missing data

| Item | Count | % |
|---|---|---|
| Speed == 0 (all 207 sensors) | 575,302 | 8.11% |
| `sensor_missing=1` (40 sensors; speed 0 -> NaN) | 107,173 | 7.82% |
| `interpolated=1` (gap <= 30 min filled) | 6,470 | 0.47% |
| Still NaN after interpolation (gap > 30 min or at series edge; excluded) | 100,703 | 7.35% |
| `weather_available=0` timestamps | 0 | 0.00% |
| Weather `temperature_2m` still NaN after <=1 h ffill | - | 0.00% |
| Weather `precipitation` still NaN after <=1 h ffill | - | 0.00% |
| Weather `relative_humidity_2m` still NaN after <=1 h ffill | - | 0.00% |

Speed 0 is treated as a sensor failure, never as a real zero. Rows with any NaN feature or target (lags/rolling windows touching a long gap) are dropped before modelling.

## Modelling table

- Rows before dropping: 1,370,880; kept: 1,237,978 (90.31%)

| Split | Rows | Low | Moderate | Severe |
|---|---|---|---|---|
| train | 880,437 | 85.1% | 7.7% | 7.2% |
| val | 184,544 | 84.5% | 7.2% | 8.3% |
| test | 172,519 | 83.2% | 7.8% | 9.0% |
