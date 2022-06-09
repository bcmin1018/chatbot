from kobert_transformers import get_tokenizer, get_kobert_model
from models.intent.IntentModel import IntentModel

bert_model = get_kobert_model()
tokenizer = get_tokenizer()

MODEL_PATH = "/models/intent/intent_class.pt"

query = "오늘 탕수육 주문 가능한가요?"
predict_model = IntentModel(bert_model, MODEL_PATH)
predict = predict_model.predict_class(query)
predict_label = predict_model.labels[predict]

print(query)
print("의도 예측 클래스 : ", predict)
print("의도 예측 레이블 : ", predict_label)