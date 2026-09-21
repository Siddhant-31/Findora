import pandas as pd
import glob
import math
import os
import json
import logging
import re
import secrets
from typing import List, Dict, Any, Optional

import google.generativeai as genai
from neo4j import GraphDatabase
from neo4j.exceptions import ConstraintError
from sentence_transformers import SentenceTransformer
from werkzeug.security import generate_password_hash, check_password_hash

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  
EMBEDDING_DIM = 384
VECTOR_INDEX_NAME = "product_embedding_index"

class KGRecommenderEngine:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password), connection_timeout=300, max_connection_lifetime=3600, connection_acquisition_timeout=600,)
        logger.info("Loading embedding model '%s'...", EMBEDDING_MODEL_NAME)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.link_lookup = {}
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.llm = genai.GenerativeModel("gemini-2.5-flash")
        logger.info("Embedding model ready.")

    def close(self):
        self.driver.close()
    
    def load_link_lookup(self, csv_folder: str):
        """Build an in-memory {product_name: link} map straight from the CSVs."""
        csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
        lookup = {}
        for file in csv_files:
            df = pd.read_csv(file).fillna("")
            for _, row in df.iterrows():
                name = str(row.get("name", "")).strip()
                link = str(row.get("link", row.get("url", row.get("product_link", "")))).strip()
                if name and link:
                    lookup[name] = link
        self.link_lookup = lookup
        logger.info("Loaded %d product links from CSV for lookup.", len(lookup))
    
    def embed_text(self, text: str) -> List[float]:
        vector = self.embedder.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def wipe_database(self):
        with self.driver.session() as session:
            while True:
                result = session.run("""
                    MATCH (n)
                    WITH n LIMIT 5000
                    DETACH DELETE n
                RETURN count(n) AS deleted
                """)
                deleted = result.single()["deleted"]
                if deleted == 0:
                    break
        logger.info("Database wiped.")

    def create_constraints(self):
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT product_id IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE")
            session.run("CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE")
            session.run("CREATE CONSTRAINT brand_name IF NOT EXISTS FOR (b:Brand) REQUIRE b.name IS UNIQUE")
            session.run("CREATE CONSTRAINT tag_name IF NOT EXISTS FOR (t:Tag) REQUIRE t.name IS UNIQUE")
            session.run("CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
            session.run("CREATE CONSTRAINT user_email IF NOT EXISTS FOR (u:User) REQUIRE u.email IS UNIQUE")
        logger.info("Constraints ensured.")

    def ensure_auth_constraints(self):
        """Safe to call on every boot: makes sure the User constraints exist even
        if the DB was seeded before auth was added, without touching product data."""
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE")
            session.run("CREATE CONSTRAINT user_email IF NOT EXISTS FOR (u:User) REQUIRE u.email IS UNIQUE")

    def load_csv_folder(self, folder):
        csv_files = glob.glob(os.path.join(folder, "*.csv"))
        logger.info("Found %d CSV files.", len(csv_files))
        BATCH_SIZE = 250
        product_id = 1

        for file in csv_files:
            logger.info("Reading %s", file)
            df = pd.read_csv(file).fillna("")
            products = []

            for _, row in df.iterrows():
                name = str(row.get("name", ""))
                category = str(row.get("sub_category", row.get("category", "Unknown")))
                brand = str(row.get("brand", "Unknown"))
                link = str(row.get("link", row.get("url", row.get("product_link", "")))).strip()
                
                txt = str(row.get("discount_price", row.get("actual_price", "")))
                txt = re.sub(r"[^\d.]", "", txt)
                try:
                    price = float(txt)
                except:
                    price = 0

                description = name
                text = f"{name}. {description}. Category: {category}. Brand: {brand}."

                products.append({
                    "id": product_id, "name": name, "category": category,
                    "brand": brand, "price": price, "description": description,
                    "link": link, "text": text
                })
                product_id += 1

            for i in range(0, len(products), BATCH_SIZE):
                batch = products[i:i+BATCH_SIZE]
                texts = [p["text"] for p in batch]
                embeddings = self.embedder.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
                rows = []

                for p, emb in zip(batch, embeddings):
                    rows.append({
                        "id": p["id"], "name": p["name"], "price": p["price"],
                        "description": p["description"], "embedding": emb.tolist(),
                        "category": p["category"], "brand": p["brand"], "link": p["link"]
                    })

                with self.driver.session() as session:
                    session.run("""
                    UNWIND $rows AS row
                    MERGE (prod:Product {id: row.id})
                    SET prod.name = row.name, prod.price = row.price, prod.description = row.description,
                        prod.embedding = row.embedding, prod.link = row.link
                    MERGE (cat:Category {name: row.category})
                    MERGE (prod)-[:BELONGS_TO]->(cat)
                    MERGE (brand:Brand {name: row.brand})
                    MERGE (prod)-[:MADE_BY]->(brand)
                    """, rows=rows)
            logger.info("%s completed", file)
        logger.info("CSV loading complete.")

    def create_vector_index(self):
        with self.driver.session() as session:
            session.run(f"""
                CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS FOR (p:Product) ON (p.embedding)
                OPTIONS {{ indexConfig: {{ `vector.dimensions`: {EMBEDDING_DIM}, `vector.similarity_function`: 'cosine' }} }}
            """)
        logger.info("Vector index '%s' ensured.", VECTOR_INDEX_NAME)

    def seed(self, data_folder: str, wipe: bool = True):
        if wipe: self.wipe_database()
        self.create_constraints()
        self.load_csv_folder(data_folder)
        self.create_vector_index()
        logger.info("Seeding complete.")
    


    def user_exists(self, email: str) -> bool:
        with self.driver.session() as session:
            rec = session.run(
                "MATCH (u:User {email: $email}) RETURN u.id AS id LIMIT 1",
                email=email,
            ).single()
        return rec is not None

    def create_user(self, email: str, password: str) -> Dict[str, Any]:
        user_id = secrets.token_hex(8)
        password_hash = generate_password_hash(password)
        token = secrets.token_urlsafe(32)
        try:
            with self.driver.session() as session:
                session.run(
                    """
                    CREATE (u:User {
                        id: $id, email: $email, password_hash: $password_hash,
                        token: $token, created_at: datetime()
                    })
                    """,
                    id=user_id, email=email, password_hash=password_hash, token=token,
                )
        except ConstraintError:
            raise ValueError("An account with this email already exists.")
        return {"user_id": user_id, "email": email, "token": token}

    def authenticate_user(self, email: str, password: str) -> Dict[str, Any]:
        with self.driver.session() as session:
            rec = session.run(
                "MATCH (u:User {email: $email}) RETURN u.id AS id, u.password_hash AS password_hash",
                email=email,
            ).single()

        if not rec or not check_password_hash(rec["password_hash"], password):
            raise ValueError("Incorrect email or password.")

        token = secrets.token_urlsafe(32)
        with self.driver.session() as session:
            session.run(
                "MATCH (u:User {id: $id}) SET u.token = $token",
                id=rec["id"], token=token,
            )
        return {"user_id": rec["id"], "email": email, "token": token}

    def get_user_by_token(self, token: str) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        with self.driver.session() as session:
            rec = session.run(
                "MATCH (u:User {token: $token}) RETURN u.id AS id, u.email AS email",
                token=token,
            ).single()
        return dict(rec) if rec else None

    def invalidate_token(self, user_id: str):
        with self.driver.session() as session:
            session.run("MATCH (u:User {id: $id}) SET u.token = null", id=user_id)



    def log_search(self, user_id: str, query: str, recommended_products: List[str] = None):
        """Records a search AND which products were recommended for it, linked to
        this user in the graph. Uses MERGE on User so this works even if the user
        node doesn't already exist (the previous MATCH-only version silently did
        nothing for users that were never explicitly created elsewhere)."""
        with self.driver.session() as session:
            session.run(
                """
                MERGE (u:User {id: $user_id})
                CREATE (u)-[:SEARCHED]->(q:SearchQuery {query: $query, timestamp: datetime()})
                WITH q
                UNWIND $recommended_products AS product_name
                MATCH (p:Product {name: product_name})
                CREATE (q)-[:RECOMMENDED]->(p)
                """,
                user_id=user_id, query=query,
                recommended_products=recommended_products or [],
            )

    def get_search_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self.driver.session() as session:
            recs = session.run(
                """
                MATCH (u:User {id: $user_id})-[:SEARCHED]->(q:SearchQuery)
                OPTIONAL MATCH (q)-[:RECOMMENDED]->(p:Product)
                WITH q, collect(p.name) AS recommended
                RETURN q.query AS query, toString(q.timestamp) AS timestamp, recommended
                ORDER BY q.timestamp DESC
                LIMIT $limit
                """,
                user_id=user_id, limit=limit,
            ).data()
        return recs

    @staticmethod
    def parse_budget(query: str):
        q = query.lower().replace(",", "")
        lakh_match = re.search(r"(\d+(\.\d+)?)\s*lakh", q)
        if lakh_match: return float(lakh_match.group(1)) * 100000
        k_match = re.search(r"(\d+(\.\d+)?)\s*k\b", q)
        if k_match: return float(k_match.group(1)) * 1000
        rupee_match = re.search(r"(?:₹|rs\.?|inr)\s*(\d+)", q)
        if rupee_match: return float(rupee_match.group(1))
        plain_match = re.search(r"\b(\d{4,7})\b", q)
        if plain_match: return float(plain_match.group(1))
        return None


    def _query_user_memory(self, user_id: str, user_query: str) -> str:
        """Neo4j-native replacement for GBrain: builds a short memory context
        from this user's past searches and what was recommended, straight from
        the graph. Works identically whether running locally or on any deployed
        instance, since it's backed by Aura rather than a local file."""
        try:
            history = self.get_search_history(user_id, limit=5)
            if not history:
                return "No prior user memory context found."

            lines = ["Recent search history for this user:"]
            for h in history:
                recommended = ", ".join(h["recommended"][:3]) if h["recommended"] else "no products matched"
                lines.append(f"- Searched '{h['query']}' -> recommended: {recommended}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning("User memory lookup failed. Error: %s", e)
            return "No prior user memory context found."


    def generate_recommendation(self, query: str, products: List[Dict], memory_context: str):
        context = ""
        for p in products:
            context += f"""
                Product: {p['name']}
                Price: ₹{p['price']}
                Brand: {p['brand']}
                Category: {p['category']}
                Description: {p['description']}
            \n"""

        prompt = f"""
    You are an advanced, personalized e-commerce shopping assistant with access to historical customer memories.

    User Profile / History Memory (From Neo4j Search History):
    {memory_context}

    Current User Query:
    {query}

    Available Products matching query parameters (From Neo4j Catalog):
    {context}

    Task:
    Recommend the best products from the catalog. Synthesize your answer by respecting their historic preferences or dislikes mentioned in the User Profile Memory. Explain clearly why these items match both their current request and past behaviors.
    """
        response = self.llm.generate_content(prompt)
        return response.text

    def retrieve_vector_only(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Pure semantic retrieval: vector similarity search against the Neo4j
        vector index only — no graph traversal for category/brand, no budget
        filtering. This is the 'vector-only RAG' baseline used for eval
        comparisons against the hybrid path below."""
        query_vector = self.embed_text(query)
        with self.driver.session() as session:
            vector_results = session.run(
                f"""
                CALL db.index.vector.queryNodes($index_name, $k, $query_vector)
                YIELD node, score
                RETURN node.id AS id, node.name AS name, node.price AS price,
                       node.description AS description, node.link AS link, score
                ORDER BY score DESC
                """,
                index_name=VECTOR_INDEX_NAME, k=top_k, query_vector=query_vector,
            ).data()

        results = []
        for r in vector_results:
            item = dict(r)
            item["price"] = item["price"] if (item["price"] and not math.isnan(item["price"])) else 0
            item["link"] = item.get("link") or self.link_lookup.get(item["name"], "")
            item["category"] = None
            item["brand"] = None
            results.append(item)
        return results

    def retrieve_hybrid(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Findora's production retrieval path: vector search over-fetches a
        wider candidate pool, each candidate is enriched with its Category/Brand
        via graph traversal, then results are filtered/re-ranked against any
        budget parsed from the query. Factored out of recommend() so the exact
        same code path used in production can be called directly for eval."""
        query_vector = self.embed_text(query)
        budget = self.parse_budget(query)

        with self.driver.session() as session:
            vector_results = session.run(
                f"""
                CALL db.index.vector.queryNodes($index_name, $k, $query_vector)
                YIELD node, score
                RETURN node.id AS id, node.name AS name, node.price AS price,
                       node.description AS description, node.specs AS specs,
                       node.link AS link, score
                ORDER BY score DESC
                """,
                index_name=VECTOR_INDEX_NAME, k=max(top_k * 3, 15), query_vector=query_vector,
            ).data()

            if not vector_results:
                return []

            enriched = []
            for r in vector_results:
                graph_context = session.run(
                    """
                    MATCH (p:Product {id: $id})-[:BELONGS_TO]->(c:Category)
                    MATCH (p)-[:MADE_BY]->(b:Brand)
                    RETURN c.name AS category, b.name AS brand
                    """, id=r["id"],
                ).single()

                item = dict(r)
                item["price"] = item["price"] if (item["price"] and not math.isnan(item["price"])) else 0
                item["link"] = item.get("link") or self.link_lookup.get(item["name"], "")
                item["category"] = graph_context["category"] if graph_context else None
                item["brand"] = graph_context["brand"] if graph_context else None
                enriched.append(item)

        if budget:
            within_budget = [p for p in enriched if p["price"] <= budget]
            results = within_budget if within_budget else sorted(enriched, key=lambda p: abs(p["price"] - budget))
        else:
            results = enriched

        return results[:top_k]

    def recommend(self, user_id: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        # 1. Read this user's memory straight from the graph (Aura-backed, works
        # identically locally or on any deployed instance).
        memory_context = self._query_user_memory(user_id, query)
        logger.info("User memory context loaded from Neo4j.")

        final_results = self.retrieve_hybrid(query, top_k=top_k)

        if not final_results:
            return {"products": [], "recommendation": "No matching products found."}

        try:
            recommendation = self.generate_recommendation(query, final_results, memory_context)
        except Exception as e:
            logger.error("LLM processing error: %s", e)
            recommendation = "Error preparing recommendation."

        try:
            # 2. Write this search + what was recommended back into the graph,
            # in one call (query text and recommended products together).
            self.log_search(user_id, query, recommended_products=[p["name"] for p in final_results])
        except Exception as e:
            logger.warning("Could not log search history for user %s: %s", user_id, e)

        return {
            "products": final_results,
            "recommendation": recommendation
        }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")

    engine = KGRecommenderEngine(uri, user, password)
    try:
        res = engine.recommend(user_id="customer_101", query="Show me running shoes under 8000")
        print("\n--- Recommendation Output ---\n", res["recommendation"])
    finally:
        engine.close()