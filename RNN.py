import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns


# === Load Dataset ===
df = pd.read_csv('final_dataset.csv')

# === Sort by ID and Date ===
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['id', 'date']).reset_index(drop=True)

# === Feature Columns and Target ===
features = [
    'activity_avg', 'appCat.builtin_avg', 'appCat.communication_used',
    'appCat.entertainment_avg', 'appCat.finance_used', 'appCat.game_avg',
    'appCat.office_used', 'appCat.social_avg', 'appCat.travel_used',
    'appCat.utilities_used', 'appCat.weather_used', 'appCat.merged_other_unknown_avg',
    'screen_max', 'sms_call_sum', 'arousal_avg', 'valence_avg'
]
target = 'mood'

#Convert Mood to Classes
def mood_to_class(mood):
    if mood < 6.8:
        return 0  # low
    elif mood <= 7.25:
        return 1  # medium
    else:
        return 2  # high
df['mood_class'] = df['mood'].apply(mood_to_class)

#Create Sequences
def create_sequences(dataframe, feature_list, target_col, seq_len=3):
    sequences = []
    targets = []
    grouped = dataframe.groupby('id')
    
    for _, group in grouped:
        group = group.reset_index(drop=True)
        for i in range(len(group) - seq_len):
            seq_x = group[feature_list].iloc[i:i+seq_len].values
            seq_y = group[target_col].iloc[i + seq_len]
            sequences.append(seq_x)
            targets.append(seq_y)
    
    return np.array(sequences), np.array(targets)

X, y = create_sequences(df, features, 'mood_class', seq_len=3)

#One-Hot Encode Labels
y_cat = to_categorical(y, num_classes=3)

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

#Loss Function
def weighted_categorical_loss(y_true, y_pred):
    true_classes = K.argmax(y_true, axis=-1)
    pred_classes = K.argmax(y_pred, axis=-1)
    class_distance = K.abs(true_classes - pred_classes)
    class_distance = K.cast(class_distance, dtype='float32')
    base_loss = K.categorical_crossentropy(y_true, y_pred)
    return base_loss * (class_distance + 1.0)

#Define the RNN Model
model = Sequential([
    SimpleRNN(64, input_shape=(3, len(features)), activation='tanh'),
    Dense(3, activation='softmax')  # 3 classes
])

model.compile(optimizer='adam',loss=weighted_categorical_loss, metrics=['accuracy'])

#Train the Model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop]
)
#Predict Classes
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

#Classification Report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=['Low', 'Medium', 'High']))

#Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix')
plt.show()
