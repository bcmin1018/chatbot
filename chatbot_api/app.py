from flask import Flask, request, jsonify, abort
import socket
import json

host = "127.0.0.1"
port = 5050

app = Flask(__name__)

def get_answer_from_engine(bottype, query):
    mySocket = socket.socket()
    mySocket.connect((host, port))

    json_data = {
        'Query': query,
        'BotType': bottype
    }
    message = json.dums(json_data)
    mySocket.send(message.encode())

    data = mySocket.recv(2048).decode()
    ret_data = json.loads(data)
    mySocket.close()
    return ret_data

@app.route('/query/<bot_type>', methods=['POST'])
def query(bot_type):
    body = request.get_json()

    try:
        if bot_type == 'TEST':
            ret = get_answer_from_engine(bottype=bot_type, query=body['query'])

        elif bot_type == "KAKAO":
            pass
    except Exception as ex:
        abort(500)

if __name__ == '__main__':
    app.run()