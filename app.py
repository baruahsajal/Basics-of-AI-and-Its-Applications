from flask import Flask, render_template, request, jsonify
from chatbot import AIChatbot
import os

app = Flask(__name__)
chatbot = AIChatbot()

if not os.path.exists('chatbot_model.h5'):
    chatbot.build_model()

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_message = request.json['message']
    intents_list = chatbot.predict_class(user_message)
    response = chatbot.get_response(intents_list)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
