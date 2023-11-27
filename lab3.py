import requests
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
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
lemmatizer = WordNetLemmatizer()

#Створення функції для опрацювання тексту згідно заданих умов
def preprocess_document(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    tokens = tokenizer.tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return ' '.join(lemmatized_tokens)

# Попередня обробка тексту
processed_text = preprocess_document(book_text)

# Виведення опрацьованого тексту
print("Опрацьований текст :")
console_width = 80 #обмеження ширини консолі, для зручного читання тексту
text_lines = textwrap.wrap(processed_text, width=console_width)
for line in text_lines[:50]:
    print(line)

print("\n" + "-"*50 + "\n")



text = processed_text
# Видалення тексту після "THE END"
positions = [match.start() for match in re.finditer(r'\bend project\b', text)]

if len(positions) >= 0:
    text = text[:positions[0]]
else:
    print(" Не виявлено 'END'")

# Видалення тексту до "Сhapter 1"
positions = [match.start() for match in re.finditer(r'\bchapter\b', text)]
if len(positions) >= 2:
    text = text[positions[12]:]
else:
    print(" Не виявлено другого згадування 'CHAPTER I'")

# Розділення тексту на глави
chapters = re.split(r'\bchapter [ivx]+\b', text)
chapters = [re.sub(r'^\s*\.+\s*', '', chapter) for chapter in chapters if chapter.strip()]

# Обробка кожної глави
processed_chapters = chapters#[preprocess_document(chapter) for chapter in chapters]

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

# Використання CountVectorizer
vectorizer = CountVectorizer(max_df=0.90, min_df=2)
dtm = vectorizer.fit_transform(chapters)

# LDA
lda_model = LatentDirichletAllocation(n_components=len(chapters), random_state=42)
lda_model.fit(dtm)

# Функція для виведення тем з LDA
def print_lda_topics(model, vectorizer, n_words=20):
    words = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(model.components_):
        print(f"Chapter {idx+1} (LDA):", [words[i] for i in topic.argsort()[-n_words:][::-1]])

print_lda_topics(lda_model, vectorizer)
#  Порівняння з TF-IDF
print("\nПорівняння результатів LDA і TF-IDF:")
for idx in range(len(chapters)):
    print(f"Chapter {idx + 1} TF-IDF: {', '.join(top_words[f'Chapter {idx + 1}'])}")
    print(f"Chapter {idx + 1} LDA: {', '.join([vectorizer.get_feature_names_out()[i] for i in lda_model.components_[idx].argsort()[-20:][::-1]])}")
    print("-" * 50)
