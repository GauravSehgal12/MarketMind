from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    BigInteger,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class StockFeature(Base):
    __tablename__ = "stock_features"

    id = Column(Integer, primary_key=True, autoincrement=True)

    symbol = Column(String(20), nullable=False, index=True)

    timestamp = Column(DateTime, nullable=False, index=True)

    close = Column(Float, nullable=False)

    sma_20 = Column(Float)
    sma_50 = Column(Float)
    ema_20 = Column(Float)

    rsi_14 = Column(Float)

    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)

    atr_14 = Column(Float)

    volatility_20 = Column(Float)

    return_1d = Column(Float)
    return_5d = Column(Float)

    volume_change = Column(Float)

    next_day_return = Column(Float)
    next_day_direction = Column(Integer)

    __table_args__ = (
        UniqueConstraint(
            "symbol",
            "timestamp",
            name="uq_feature_symbol_timestamp",
        ),
    )