from sqlalchemy import create_engine
from config.setting import POSTGRES_URL

engine = create_engine(POSTGRES_URL)
