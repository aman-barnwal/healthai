from flask import Blueprint, request, jsonify

from backend.ai.assistant import chat_with_healthcare_ai


chat_bp = Blueprint(
    "chat",
    __name__
)


@chat_bp.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        message = str(
            data.get(
                "message",
                ""
            )
        ).strip()

        if not message:

            return jsonify({
                "success": False,
                "error": "Message is required."
            }), 400


        # ---------------------------------------------
        # SEND MESSAGE TO HYBRID AI ENGINE
        # ---------------------------------------------

        reply = chat_with_healthcare_ai(
            message
        )


        # ---------------------------------------------
        # RETURN RESPONSE TO FRONTEND
        # ---------------------------------------------

        return jsonify({

            "success": True,

            "message": message,

            "reply": reply

        })


    except Exception as error:

        print(
            "[HealthcareAI API Error]:",
            error
        )

        return jsonify({

            "success": False,

            "error": "Something went wrong while processing your request."

        }), 500
