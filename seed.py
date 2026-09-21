from dotenv import load_dotenv
load_dotenv(override = True)
import os
from kg_engine import KGRecommenderEngine

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USER")
password = os.getenv("NEO4J_PASSWORD")
csv_folder = os.getenv("CSV_FOLDER", "data")

engine = KGRecommenderEngine(uri, user, password)
engine.seed(csv_folder, wipe=False)
engine.close()