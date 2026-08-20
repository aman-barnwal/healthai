from flask import Flask, jsonify
from flask_cors import CORS


def create_app():
    app = Flask(__name__)

    # Allow frontend applications to communicate with the backend.
    CORS(app)

    # ------------------------------------------------------------
    # BASIC HEALTH CHECK
    # ------------------------------------------------------------

    @app.route("/", methods=["GET"])
    def home():
        return jsonify({
            "status": "online",
            "service": "HealthcareAI API",
            "message": "HealthcareAI backend is running."
        })


    @app.route("/api/health", methods=["GET"])
    def health_check():
        return jsonify({
            "status": "healthy"
        })


    # ------------------------------------------------------------
    # ROUTES
    # ------------------------------------------------------------

    from backend.routes.chat import chat_bp

    app.register_blueprint(
        chat_bp,
        url_prefix="/api"
    )

    return app


# ------------------------------------------------------------
# RUN SERVER
# ------------------------------------------------------------

if __name__ == "__main__":

    app = create_app()

    print("\nHealthcareAI API starting...")
    print("API running at: http://127.0.0.1:5000")
    print("Health check: http://127.0.0.1:5000/api/health\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
