import joblib
import numpy as np

model = joblib.load("model_knn.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")


def prediksi(data):

    data = np.array(data).reshape(1, -1)

    data = scaler.transform(data)

    hasil = model.predict(data)

    probabilitas = model.predict_proba(data)

    kelas = label_encoder.inverse_transform(hasil)[0]

    confidence = probabilitas.max() * 100

    return kelas, confidence