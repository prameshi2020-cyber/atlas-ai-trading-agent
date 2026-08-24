import sqlite3

c = sqlite3.connect('data/atlas.db')
print(c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall())
print(c.execute("SELECT symbol, COUNT(*) FROM market_ticks GROUP BY symbol").fetchall())
c.close()
