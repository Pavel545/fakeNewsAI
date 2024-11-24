import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import joblib

# Загрузка данных
df = pd.read_csv('./fake_news.csv')

# Просмотр первых нескольких строк данных
print(df.head())

# Проверка на наличие пропущенных значений
print(df.isnull().sum())

# Разделим данные на признаки (X) и целевую переменную (y):
X = df['text']  # заменим 'text' на название столбца с текстом новостей
y = df['label']  # заменим 'label' на название столбца с метками

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Дополнительно выделяем часть обучающего массива для тестирования
X_train_part, X_test_part, y_train_part, y_test_part = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Используем TfidfVectorizer для преобразования текстовых данных в векторы:
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)  # Обучаем векторизатор на всей обучающей выборке
X_test_tfidf = tfidf_vectorizer.transform(X_test)  # Преобразуем тестовую выборку

# Создаем и обучаем классификатор PassiveAggressiveClassifier:
classifier = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
classifier.fit(X_train_tfidf, y_train)  # Обучаем модель на всей обучающей выборке

# Сделаем предсказания и оцените точность модели:
y_pred = classifier.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Создаем матрицу ошибок и визуализируем её:
confusion_mat = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Сохраняем модели 
joblib.dump(classifier, 'fake_news_classifier.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Затем загружаем после обучения 
loaded_classifier = joblib.load('fake_news_classifier.pkl')
loaded_tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Векторизация тестовой части обучающего массива
X_test_tfidf_part = loaded_tfidf_vectorizer.transform(X_test_part)

# Предсказание
predictions_part = loaded_classifier.predict(X_test_tfidf_part)

# Визуализация результатов
labels = np.unique(y_test_part)  # Получаем уникальные классы
counts = [np.sum(predictions_part == label) for label in labels]  # Подсчет предсказаний по классам

# Построение столбчатой диаграммы
plt.bar(labels, counts, color=['blue', 'orange'])
plt.xlabel('Классы')
plt.ylabel('Количество предсказаний')
plt.title('Распределение предсказанных классов')
plt.xticks(labels, ['Real', 'Fake'])  # Переименуйте классы по необходимости
plt.show()

# Дополнительно: Матрица путаницы
cm = confusion_matrix(y_test_part, predictions_part, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Матрица путаницы')
plt.show()


