# fake_news_classifier.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

class Model:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer()
        # можно сразу 1000 но сделаю по 1 чтобы красива наблюдать прогрес обучения? 
        # ! как в итоге оказалось бред, ибо со старта 98%, мб тестовый фаил такой простой но разницы нет, что по ипохам расписывать что нет
        self.classifier = PassiveAggressiveClassifier(max_iter=1, random_state=42)
    
    def load_data(self, filePath):
        df = pd.read_csv(filePath)
        print(df.head())
        # есть ли пустые и сколько
        print(df.isnull().sum())
        # отбирем контент и значение 
        self.x=df['text']
        self.y=df['label']
    
    def slit_data(self):
        # отбираем пропорцианально делим 80 к 20
        x_train, x_test, y_train, y_test = train_test_split(self.x,self.y, test_size=0.2, random_state=42)
        # тренировочные записсываем 
        self.x_train = x_train
        self.y_train = y_train
        # отдаём тестовые
        return  x_test, y_test
    
    def train(self):
            # Обучаем векторизатор на всей обучающей выборке
            x_v = self.vectorizer.fit_transform(self.x_train)

            # Обучаем модель на всей обучающей выборке
            n_epoch = 5

            train_progress = []
            for n in range(n_epoch):
                # Обучаем модель на одной эпохе
                self.classifier.partial_fit(x_v, self.y_train, classes=np.unique(self.y))
                # Оценка точности на обучающих данных
                x_trainVect = self.vectorizer.transform(self.x_train)
                y_train_pred = self.classifier.predict(x_trainVect)
                train_accuracy = accuracy_score(self.y_train, y_train_pred)
                train_progress.append(train_accuracy)  # Сохраняем точность на обучающих данных
                print(f"Эпоха {n + 1}/{n_epoch}, Точность: {train_accuracy:.2f}")
            # и красивенько выводим графиком прогрессию обучения нашей модели
            # можно канечно обойтись и без прогрессии что избавит нас от строчек кода благо класс позволяет
            # но нет
            self.pltProgress(n_epoch,train_progress)
            # сохраняем хз нужно ли
            self.save()
    
    def pltProgress(self,n_epoch,train_progress):
        # Построение графика
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, n_epoch + 1), train_progress, marker='o', linestyle='-', color='b')
        plt.title('Прогресс обучения по эпохам (Точность на обучающих данных)')
        plt.xlabel('Эпоха')
        plt.ylabel('Точность')
        plt.xticks(np.arange(1, n_epoch + 1, step=5))  # Устанавливаем шаг по оси X
        plt.ylim(0, 1)  # Устанавливаем пределы по оси Y
        plt.grid()
        plt.show()

    def save(self):
        joblib.dump(self.classifier, 'fake_news_classifier.pkl')
        joblib.dump(self.vectorizer, 'tfidf_vectorizer.pkl')

    def load(self):
        loaded_classifier = joblib.load('fake_news_classifier.pkl')
        loaded_tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return loaded_classifier, loaded_tfidf_vectorizer
    
    def test(self, X, Y):
        loaded_classifier, loaded_tfidf_vectorizer =self.load()

        X_test = loaded_tfidf_vectorizer.transform(X)
        # Предсказание
        predictions = loaded_classifier.predict(X_test)

        # Визуализация результатов
        labels = np.unique(Y)  # Получаем уникальные классы
        counts = [np.sum(predictions == label) for label in labels]  # Подсчет предсказаний по классам

        # Построение столбчатой диаграммы
        plt.bar(labels, counts, color=['blue', 'orange'])
        plt.xlabel('Классы')
        plt.ylabel('Количество предсказаний')
        plt.title('Распределение предсказанных классов')
        plt.xticks(labels, ['Real', 'Fake'])  # Переименуйте классы по необходимости
        plt.show()




        




    



# class FakeNewsClassifier:

#     def load_data(self, filepath):
#         df = pd.read_csv(filepath)
#         print(df.head())
#         print(df.isnull().sum())
#         self.X = df['text']  
#         self.y = df['label'] 

#     def split_data(self):
#         X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
#         self.X_train_part, self.X_test_part, self.y_train_part, self.y_test_part = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
#         return X_train, X_test, y_train, y_test

#     def vectorize_data(self, X_train, X_test):
#         self.X_train_tfidf = self.vectorizer.fit_transform(X_train)
#         self.X_test_tfidf = self.vectorizer.transform(X_test)

#     def train(self):
#         print(f"X_train_tfidf shape: {self.X_train_tfidf.shape}")
#         print(f"y_train_part shape: {self.y_train_part.shape}")
#         self.classifier.fit(self.X_train_tfidf, self.y_train_part)

#     def predict(self):
#         self.y_pred = self.classifier.predict(self.X_test_tfidf)
#         return self.y_pred

#     def evaluate(self, y_test):
#         accuracy = accuracy_score(y_test, self.y_pred)
#         print(f'Accuracy: {accuracy * 100:.2f}%')
#         return accuracy

#     def plot_confusion_matrix(self, y_test):
#         confusion_mat = confusion_matrix(y_test, self.y_pred)
#         plt.figure(figsize=(8, 6))
#         sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.title('Confusion Matrix')
#         plt.show()

#     def save_model(self, model_path, vectorizer_path):
#         joblib.dump(self.classifier, model_path)
#         joblib.dump(self.vectorizer, vectorizer_path)

#     def load_model(self, model_path, vectorizer_path):
#         self.classifier = joblib.load(model_path)
#         self.vectorizer = joblib.load(vectorizer_path)

#     def predict_with_loaded_model(self, X_test_part):
#         X_test_tfidf_part = self.vectorizer.transform(X_test_part)
#         predictions_part = self.classifier.predict(X_test_tfidf_part)
#         return predictions_part

#     def plot_prediction_distribution(self, predictions_part):
#         labels = np.unique(self.y_test_part)
#         counts = [np.sum(predictions_part == label) for label in labels]
#         plt.bar(labels, counts, color=['blue', 'orange'])
#         plt.xlabel('Классы')
#         plt.ylabel('Количество предсказаний')
#         plt.title('Распределение предсказанных классов')
#         plt.xticks(labels, ['Real', 'Fake'])
#         plt.show()

#     def plot_error_matrix(self, predictions_part):
#         cm = confusion_matrix(self.y_test_part, predictions_part)
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(self.y_test_part))
#         disp.plot(cmap=plt.cm.Blues)
#         plt.title('Матрица ошибок')
#         plt.show()
