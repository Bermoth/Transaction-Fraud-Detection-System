from database import engine
from models import Base


def initialize_database():
    Base.metadata.create_all(bind=engine)
    print("Database tables initialized successfully.")

    table_names = Base.metadata.tables.keys()
    print("Tables defined:", ", ".join(table_names))


if __name__ == "__main__":
    initialize_database()