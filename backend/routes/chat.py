from flask import Blueprint, request, jsonify


chat_bp = Blueprint(
    "chat",
    __name__
)


@chat_bp.route("/chat", methods=["POST"])
def chat():

    data = request.get_json(
        silent=True
    ) or {}

    message = data.get(
        "message",
        ""
    ).strip()

    if not message:

        return jsonify({
            "success": False,
            "error": "Message is required."
        }), 400


    return jsonify({
        "success": True,
        "message": message,
        "reply": "HealthcareAI chat endpoint is working."
    })
