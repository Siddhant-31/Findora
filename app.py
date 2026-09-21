import os
import logging
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from kg_engine import KGRecommenderEngine

load_dotenv(override=True)  # .env always wins, even over a pre-existing system/user env var

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))

app = Flask(__name__)

engine = None


def get_engine():
    global engine
    if engine is None:
        logger.info("Initializing KG engine and connecting to Neo4j at %s", NEO4J_URI)
        engine = KGRecommenderEngine(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        engine.ensure_auth_constraints()
        engine.load_link_lookup(os.getenv("CSV_FOLDER", "data"))  # <-- add this
    return engine


def error_response(message, code=400):
    return jsonify({"error": message}), code


def bearer_token():
    header = request.headers.get("Authorization", "")
    return header[7:].strip() if header.startswith("Bearer ") else ""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/recommend", methods=["POST"])
def recommend():
    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    

    user_id = (body.get("user_id") or "guest_user").strip()

    if not query:
        return jsonify({"error": "Please provide a query"}), 400


    results = get_engine().recommend(user_id=user_id, query=query)

    return jsonify({
        "user_id": user_id,
        "query": query,
        "count": len(results["products"]),
        "products": results["products"],
        "recommendation": results["recommendation"]
    })


@app.route("/api/auth/check", methods=["POST"])
def auth_check():
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()

    if not email:
        return error_response("Please provide an email address.")

    try:
        exists = get_engine().user_exists(email)
    except Exception:
        logger.exception("auth_check failed")
        return error_response("Could not check that email right now.", 500)

    return jsonify({"exists": exists})


@app.route("/api/auth/signup", methods=["POST"])
def auth_signup():
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""

    if not email:
        return error_response("Please provide an email address.")
    if len(password) < 8:
        return error_response("Password must be at least 8 characters.")

    try:
        result = get_engine().create_user(email, password)
    except ValueError as e:
        return error_response(str(e), 409)
    except Exception:
        logger.exception("auth_signup failed")
        return error_response("Could not create your account right now.", 500)

    logger.info("New account created: %s", email)
    return jsonify(result)


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    body = request.get_json(silent=True) or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""

    if not email or not password:
        return error_response("Please provide an email and password.")

    try:
        result = get_engine().authenticate_user(email, password)
    except ValueError as e:
        return error_response(str(e), 401)
    except Exception:
        logger.exception("auth_login failed")
        return error_response("Could not sign in right now.", 500)

    return jsonify(result)


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    body = request.get_json(silent=True) or {}
    user_id = (body.get("user_id") or "").strip()
    token = bearer_token()

    user = get_engine().get_user_by_token(token) if token else None
    if not user or user["id"] != user_id:
        return error_response("Unauthorized.", 401)

    get_engine().invalidate_token(user_id)
    return jsonify({"ok": True})


@app.route("/api/history", methods=["GET"])
def history():
    user_id = (request.args.get("user_id") or "").strip()
    token = bearer_token()

    if not user_id:
        return error_response("user_id is required.")

    user = get_engine().get_user_by_token(token) if token else None
    if not user or user["id"] != user_id:
        return error_response("Unauthorized.", 401)

    try:
        items = get_engine().get_search_history(user_id)
    except Exception:
        logger.exception("history lookup failed")
        return error_response("Could not load history right now.", 500)

    return jsonify({"history": items})


@app.route("/api/health")
def health():
    try:
        eng = get_engine()
        with eng.driver.session() as session:
            session.run("RETURN 1").consume()
        return jsonify({"status": "ok", "neo4j": "connected"})
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


if __name__ == "__main__":
    # debug=True is a real security hole in production (lets anyone execute
    # code via the browser-based debugger on exceptions). Default to off;
    # set FLASK_DEBUG=true in your LOCAL .env only, never on the deployed server.
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=debug_mode)