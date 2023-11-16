
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Зчитування даних з файлу
data = np.genfromtxt('data_banknote_authentication.txt', delimiter=',')
X = data[:, :-1]  # Параметри, які використовують для навчання моделі
y = data[:, -1]  # Результати

# Розділення даних на тренувальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic loss
def logistic_loss(y_true, y_pred):
    return tf.math.log1p(tf.exp(-y_true * y_pred))

# Binary Crossentropy
def binary_crossentropy(y_true, y_pred):
    return -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))

#Adaboost loss
def adaboost_loss(y_true, y_pred):
    return tf.exp(-y_true * y_pred)

# Функція для тренування та оцінки моделі
def train_and_evaluate(loss_function, X_train, y_train, X_test, y_test, epochs=100):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True)
    ])

    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test))

    train_loss = history.history['loss']
    test_loss = history.history['val_loss']

    return model, train_loss, test_loss

loss_functions = [logistic_loss, binary_crossentropy, adaboost_loss]

models = []
train_losses = []
test_losses = []

for loss_function in loss_functions:
    model, train_loss, test_loss = train_and_evaluate(loss_function, X_train, y_train, X_test, y_test)
    models.append(model)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

# Створення графіків  кривих навчання
plt.figure(figsize=(12, 6))
for i, loss_function in enumerate(loss_functions):
    plt.plot(train_losses[i], label=f"Train Loss ({loss_function.__name__})")
    plt.plot(test_losses[i], label=f"Test Loss ({loss_function.__name__})")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.legend()
plt.grid(True)
plt.show()

# Порівняння якості класифікації за метрикою accuracy
accuracies = []

for model in models:
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

for i, loss_function in enumerate(loss_functions):
    print(f"Accuracy ({loss_function.__name__}): {accuracies[i]}")