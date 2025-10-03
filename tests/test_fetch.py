from src.data.fetch_yfinance import fetch_gold

def test_fetch_gold():
    df = fetch_gold(start="2024-01-01", end="2024-01-10")
    assert not df.empty
    assert "Close" in df.columns