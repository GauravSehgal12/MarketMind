from .connection import engine
from .models import Base


def initialize_database():
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    initialize_database()
    print("Database tables created successfully.")