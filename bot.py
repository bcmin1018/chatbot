import threading
import json
from config.DatabaseConfig import *
from utils.Database import Database
from utils.FindAnswer import FindAnswer
from utils.BotServer import BotServer
from kobert_transformers import get_tokenizer, get_kobert_model
from models.intent.IntentModel import IntentModel

bert_model = get_kobert_model()
MODEL_PATH = "C:\\Users\\min\\PycharmProjects\\chatbot\\code\\models\\intent\\intent_class.pt"
intent = IntentModel(bert_model, MODEL_PATH)


def to_client(conn, addr, params):
    db = params['db']
    try:
        db.connect()
        read = conn.recv(2048)
        print('==================')
        print('Connection from : %s' % str(addr))

        if read is None or not read:
            print('클라이언트 연결 끊어짐')
            exit(0)

        recv_json_data = json.loads(read.decode())
        print("데이터 수신 : ", recv_json_data)
        query = recv_json_data['Query']

        intent_predict = intent.predict_class(query)
        intent_name = intent.labels[intent_predict]

        # ner_predicts = ner.predict(query)
        # ner_tags = ner.predict_tags(query)

        try:
            f = FindAnswer(db)
            answer_text, answer_image = f.search(intent_name, None)
            print(answer_text)
            print(answer_image)
            answer = f.tag_to_word(None, answer_text)

        except:
            answer = "죄송해요 무슨 말인지 모르겠어요. 조금 더 공부할게요."
            answer_image = None

        send_json_data_str = {
            "Query" : query,
            "Answer" : answer,
            "AnswerImageURL" : answer_image,
            "Intent" : intent_name,
            # "NER" : str(ner_predicts)
            "NER" : None
        }
        message = json.dumps(send_json_data_str)
        conn.send(message.encode())

    except Exception as ex:
        print(ex)

    finally:
        if db is not None:
            db.close()
        conn.close()

if __name__ == '__main__':
    db = Database(host = DB_HOST, user = DB_USER, password = DB_PASSWORD, db_name = DB_NAME)
    print('DB 접속')

    port = 5050
    listen = 100
    bot = BotServer(port, listen)
    bot.create_sock()
    print("bot start")

    while True:
        conn, addr = bot.ready_for_client()
        params = {
            "db" : db
        }
        client = threading.Thread(target=to_client, args=(conn, addr, params))
        client.start()