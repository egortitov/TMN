from gensim import corpora

import requests
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import textwrap
import string

# Завантаження тексту книги
url = "http://www.gutenberg.org/files/11/11-0.txt"
response = requests.get(url)
response.encoding = 'utf-8'
book_text = response.text

# Ініціалізація токенізатора, списку стоп-слів і лематизатора
tokenizer = nltk.WordPunctTokenizer()
stop_words = set(nltk.corpus.stopwords.words('english'))
stop_words.add("im")
stop_words.add("oh")
stop_words.add("em")

lemmatizer = WordNetLemmatizer()

#Створення функції для опрацювання тексту згідно заданих умов
def preprocess_document(text):
    # Видалення цифр та неалфавітних символів
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Приведення тексту до нижнього регістру і видалення зайвих пробілів на початку і в кінці
    text = text.lower().strip()

    tokens = tokenizer.tokenize(text)
    # Видалення стоп-слів
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Лематизація
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Повертаємо оброблений текст
    return ' '.join(lemmatized_tokens)

# Попередня обробка тексту
processed_text = preprocess_document(book_text)

# Виведення опрацьованого тексту
print("Опрацьований текст :")
console_width = 80 #обмеження ширини консолі, для зручного читання тексту
text_lines = textwrap.wrap(processed_text, width=console_width)
for line in text_lines[-30:]:
    print(line)

print("\n" + "-"*50 + "\n")

# Повернення тексту до початкового формату
text = processed_text

# Знаходження всіх позицій, де зустрічається "END"
positions = [match.start() for match in re.finditer(r'\bend project\b', text)]

if len(positions) >= 0:
    # Відсікаємо текст до  "END"
    text = text[:positions[0]]
else:
    print(" Не виявлено 'END'")

# Знаходження всіх позицій, де зустрічається "CHAPTER I"
positions = [match.start() for match in re.finditer(r'\bchapter\b', text)]
if len(positions) >= 2:
    # Відсікаємо текст до другого згадування "CHAPTER I"
    text = text[positions[12]:]
else:
    print(" Не виявлено другого згадування 'CHAPTER I'")

# Розділення тексту на глави з використанням римських чисел
chapters = re.split(r'\bchapter [ivx]+\b', text)
# Видалення порожніх розділів та зайвих символів після розділу
chapters = [re.sub(r'^\s*\.+\s*', '', chapter) for chapter in chapters if chapter.strip()]

# Обробка кожної глави
processed_chapters = [preprocess_document(chapter) for chapter in chapters]

# Видалення знаків пунктуації для кожного абзацу
for idx in range(len(processed_chapters)):
    processed_chapters[idx] = processed_chapters[idx].translate(str.maketrans('', '', string.punctuation))

# Застосування TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_df=0.85, max_features=20)
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_chapters)

# Вибір Топ-20 слів для кожної глави
top_words = {}

feature_names = tfidf_vectorizer.get_feature_names_out()
for idx, row in enumerate(tfidf_matrix):
    top_indices = row.toarray()[0].argsort()[-20:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_words[f"Chapter {idx+1}"] = top_features

# Ініціалізуйте TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')

# Побудуйте матрицю tf-idf
tfidf_matrix = tfidf_vectorizer.fit_transform(chapters)

# Отримайте словник та інвертований словник для подальшої роботи з gensim
id2word = corpora.Dictionary([ch.split() for ch in chapters])
corpus = [id2word.doc2bow(ch.split()) for ch in chapters]

# Визначте кількість тем
num_topics = 12  # Змініть це на бажану кількість тем

# Проведіть модель LDA
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda_model.fit(tfidf_matrix)

# Виведіть теми та їх ключові слова
for topic_idx, topic in enumerate(lda_model.components_):
    print(f"Topic #{topic_idx + 1}:")
    print("LDA:",end=" ")
    print([tfidf_vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-20:]])
    print("TF-IDF:",end=" ")
    print(f"{top_words[f'Chapter {topic_idx + 1}']}",end="\n\n")