import yaml
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Load configuration
config_path = "interface/config.yaml"

with open(config_path, "r") as file:
    config = yaml.safe_load(file)

SQLALCHEMY_DATABASE_URL = (
    f"postgresql://{config['database']['user']}:"
    f"{config['database']['password']}@db:5432/"
    f"{config['database']['name']}"
)

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
