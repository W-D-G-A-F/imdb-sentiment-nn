import gradio as gr
import joblib
import tensorflow as tf

vec = joblib.load('model/tfidf.pkl')
model = tf.keras.models.load_model('model/imdb_nn_model.h5')

def predict(text):
    feat = vec.transform([text]).toarray()
    score = model.predict(feat, verbose=0)[0][0]
    sentiment = "Positive" if score > 0.5 else "Negative"
    return sentiment, float(score)

demo = gr.Interface(
    fn=predict,
    inputs="text",
    outputs=["text", "number"],
    title="IMDB Sentiment Analysis"
)
demo.launch(server_name="0.0.0.0", server_port=7860)
