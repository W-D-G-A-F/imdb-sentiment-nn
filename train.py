import pandas as pd
import numpy as np
import joblib
import os  # 导入os模块创建文件夹
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 关键：先创建model文件夹
os.makedirs("model", exist_ok=True)

# 加载数据集（匹配你的CSV列名）
df = pd.read_csv("data/imdb_top_500.csv")
x = df["text"]
y = df["label"]

# TF-IDF向量化
vec = TfidfVectorizer(max_features=5000)
x_vec = vec.fit_transform(x).toarray()
joblib.dump(vec, "model/tfidf.pkl")

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x_vec, y, test_size=0.2, random_state=42)

# 搭建情感分析神经网络
model = Sequential([
    Dense(256, activation="relu", input_shape=(5000,)),
    Dropout(0.3),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=4, batch_size=64, validation_split=0.1)

# 保存模型
model.save("model/imdb_nn_model.h5")
print("✅ 模型训练完成，已保存到 model/ 文件夹")
