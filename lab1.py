import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, precision_recall_curve, \
    roc_curve

# Завантаження набору даних
data = pd.read_csv("bioresponse.csv")

# Розділення набору даних на навчальний і тестовий
X = data.drop("Activity", axis=1)
y = data["Activity"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Навчання дрібного дерева рішень
classifier_dt = DecisionTreeClassifier(criterion="entropy", max_depth=5)
classifier_dt.fit(X_train, y_train)

# Навчання глибокого дерева рішень
classifier_dt_deep = DecisionTreeClassifier(criterion="entropy", max_depth=20)
classifier_dt_deep.fit(X_train, y_train)

# Навчання випадкового лісу на дрібних деревах
classifier_rf_dt = RandomForestClassifier(n_estimators=100, max_depth=5)
classifier_rf_dt.fit(X_train, y_train)

# Навчання випадкового лісу на глибоких деревах
classifier_rf_dt_deep = RandomForestClassifier(n_estimators=100, max_depth=20)
classifier_rf_dt_deep.fit(X_train, y_train)

models = [classifier_dt, classifier_dt_deep, classifier_rf_dt, classifier_rf_dt_deep]
for model in models:
    y_pred = model.predict(X_test)
    accuracy_dt = accuracy_score(y_test, y_pred)
    precision_dt = precision_score(y_test, y_pred)
    recall_dt = recall_score(y_test, y_pred)
    f1_score_dt = f1_score(y_test, y_pred)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    log_loss_dt = log_loss(y_test, y_pred_proba)

    print("Accuracy:", accuracy_dt, end=" ")
    print("Precision:", precision_dt, end=" ")
    print("Recall:", recall_dt, end=" ")
    print("F1-score:", f1_score_dt, end=" ")
    print("log-loss:", log_loss_dt)

    # Precision-Recall і ROC-криві
    import matplotlib.pyplot as plt
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label='ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    #plt.show()

# Навчання класифікаторів які уникають помилок II роду
classifier_dt_2 = DecisionTreeClassifier(criterion="entropy", class_weight='balanced', max_depth=5)
classifier_dt_2.fit(X_train, y_train)

classifier_dt_deep_2 = DecisionTreeClassifier(criterion="entropy", class_weight='balanced', max_depth=20)
classifier_dt_deep_2.fit(X_train, y_train)

classifier_rf_dt_2 = RandomForestClassifier(n_estimators=100, class_weight='balanced',max_depth=5)
classifier_rf_dt_2.fit(X_train, y_train)

classifier_rf_dt_deep_2 = RandomForestClassifier(n_estimators=100, class_weight='balanced',max_depth=20)
classifier_rf_dt_deep_2.fit(X_train, y_train)
print("\n")

models_2 = [classifier_dt_2, classifier_dt_deep_2, classifier_rf_dt_2, classifier_rf_dt_deep_2]
for model in models_2:
    # Оцінка якості моделі
    threshold = 0.2
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > threshold).astype(int)


    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)

    print("Accuracy:", acc, end=" ")
    print("Precision:", precision, end=" ")
    print("Recall:", recall, end=" ")
    print("F1-score:", f1, end=" ")
    print("log-loss:", logloss)

