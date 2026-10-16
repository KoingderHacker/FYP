import pandas as pd

from agents.solar import forecasting as solar_forecasting


def test_open_meteo_weather_dataframe_enters_chronos_solar_pipeline(monkeypatch):
    historical = pd.read_csv(
        "data/wind/raw/gujarat_hourly_weather_generation_final.csv"
    )
    origin = pd.Timestamp("2026-10-06T00:11:00+05:30")
    future_times = pd.date_range(
        "2026-10-06T01:00:00+05:30", periods=24, freq="h"
    )
    future_weather = pd.DataFrame({
        "datetime": future_times,
        "ALLSKY_SFC_SW_DWN": [500.0] * 24,
        "CLRSKY_SFC_SW_DWN": [600.0] * 24,
        "T2M": [28.0] * 24,
        "RH2M": [70.0] * 24,
        "WS10M": [8.0] * 24,
        "WS50M": [10.0] * 24,
        "PS": [1005.0] * 24,
        "PRECTOTCORR": [0.0] * 24,
        "WD10M": [245.0] * 24,
        "WD50M": [248.0] * 24,
    })
    captured = {}

    class FakeChronos2:
        def predict_df(self, context, *, future_df, **kwargs):
            captured["context"] = context.copy()
            captured["future"] = future_df.copy()
            return pd.DataFrame({
                "datetime": future_df["datetime"],
                0.1: [1.0] * len(future_df),
                0.5: [2.0] * len(future_df),
                0.9: [3.0] * len(future_df),
            })

    monkeypatch.setattr(
        solar_forecasting, "_load_pipeline", lambda: FakeChronos2()
    )
    result = solar_forecasting.forecast_solar(
        historical,
        horizon_name="1_day",
        horizon_hours=24,
        forecast_origin=origin.isoformat(),
        future_weather_df=future_weather,
    )

    assert len(result["forecast"]) == 24
    assert result["forecast"][0]["timestamp"] == "2026-10-06T01:00:00+05:30"
    assert len(captured["context"]) == solar_forecasting.CONTEXT_LENGTH
    assert captured["context"]["Gujarat_Solar_MU_hourly_est"].notna().all()
    assert len(captured["future"]) == 24
    assert "Gujarat_Solar_MU_hourly_est" not in captured["future"].columns
    assert captured["future"]["T2M"].eq(28.0).all()
    assert captured["future"]["WS50M"].eq(10.0).all()
    assert captured["future"]["hour_sin"].notna().all()
