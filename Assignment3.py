# ---------------------------------------------------------------
# 1. Load and Preprocess Local Jena Climate Dataset
# ---------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Local dataset path
fname = r"C:\austi\School\Documents\jena_climate_2009_2016.csv"

# Verify file existence
if not os.path.exists(fname):
    raise FileNotFoundError(f"Dataset not found at {fname}")

# Read CSV content
with open(fname) as f:
    data = f.read()

lines = data.split("\n")
header = lines[0].split(",")
lines = lines[1:]
print("Header:", header)
print("Number of data rows:", len(lines))

# ---------------------------------------------------------------
# 2. Parse data into arrays
# ---------------------------------------------------------------
temperature = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    if not line.strip():
        continue
    values = [float(x) for x in line.split(",")[1:]]
    temperature[i] = values[1]  # 2nd column = T (degC)
    raw_data[i, :] = values[:]

# Quick visualizations
plt.figure(figsize=(10, 4))
plt.plot(range(len(temperature)), temperature)
plt.title("Temperature over time (Jena Climate)")
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(range(1440), temperature[:1440])
plt.title("Temperature (First 1440 samples)")
plt.show()

# ---------------------------------------------------------------
# 3. Train/validation/test split
# ---------------------------------------------------------------
num_train_samples = int(0.5 * len(raw_data))
num_val_samples = int(0.25 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples
print("num_train_samples:", num_train_samples)
print("num_val_samples:", num_val_samples)
print("num_test_samples:", num_test_samples)

# Normalize using training data stats
mean = raw_data[:num_train_samples].mean(axis=0)
std = raw_data[:num_train_samples].std(axis=0)
raw_data -= mean
raw_data /= std

# ---------------------------------------------------------------
# 4. Create Time Series Datasets
# ---------------------------------------------------------------
sampling_rate = 6
sequence_length = 120
delay = sampling_rate * (sequence_length + 24 - 1)
batch_size = 256

# Check for NaNs/Infs and fix them
print("NaNs in raw_data:", np.isnan(raw_data).sum())
print("Infs in raw_data:", np.isinf(raw_data).sum())
raw_data = np.nan_to_num(raw_data, nan=0.0, posinf=0.0, neginf=0.0)
temperature = np.nan_to_num(temperature, nan=0.0, posinf=0.0, neginf=0.0)

train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data,
    targets=temperature,
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples - delay
)

val_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data,
    targets=temperature,
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples - delay
)

test_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data,
    targets=temperature,
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples,
    end_index=None
)

for samples, targets in train_dataset:
    print("samples shape:", samples.shape)
    print("targets shape:", targets.shape)
    break

# ---------------------------------------------------------------
# 5. Baseline (Naive) Evaluation
# ---------------------------------------------------------------
def evaluate_naive_method(dataset):
    total_abs_err = 0.0
    samples_seen = 0
    for samples, targets in dataset:
        preds = samples[:, -1, 1] * std[1] + mean[1]
        total_abs_err += np.sum(np.abs(preds - targets))
        samples_seen += samples.shape[0]
    return total_abs_err / samples_seen

print(f"Validation MAE: {evaluate_naive_method(val_dataset):.2f}")
print(f"Test MAE: {evaluate_naive_method(test_dataset):.2f}")

# ---------------------------------------------------------------
# 6. Dense (Fully Connected) Model
# ---------------------------------------------------------------
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.Flatten()(inputs)
x = layers.Dense(16, activation="relu")(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

callbacks = [
    keras.callbacks.ModelCheckpoint("jena_dense.keras", save_best_only=True)
]

model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(
    train_dataset,
    epochs=15,
    validation_data=val_dataset,
    callbacks=callbacks
)

model = keras.models.load_model("jena_dense.keras")
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")

plt.figure()
plt.plot(history.history["mae"], "bo", label="Training MAE")
plt.plot(history.history["val_mae"], "b", label="Validation MAE")
plt.title("Training and Validation MAE (Dense Model)")
plt.legend()
plt.show()

# ---------------------------------------------------------------
# 7. LSTM Model
# ---------------------------------------------------------------
from tensorflow.keras.layers import LSTM, Dropout

inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = LSTM(64, return_sequences=True)(inputs)
x = Dropout(0.2)(x)
x = LSTM(64)(x)
x = Dropout(0.2)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(
    train_dataset,
    epochs=15,
    validation_data=val_dataset,
    callbacks=callbacks
)

test_mae = model.evaluate(test_dataset)[1]
print(f"Test MAE: {test_mae:.2f}")

plt.figure()
plt.plot(history.history["mae"], "bo", label="Training MAE")
plt.plot(history.history["val_mae"], "b", label="Validation MAE")
plt.title("Training and Validation MAE (LSTM)")
plt.legend()
plt.show()

# ---------------------------------------------------------------
# 8. Deeper LSTM Model
# ---------------------------------------------------------------
model = keras.Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(sequence_length, raw_data.shape[-1])))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(layers.Dense(1))

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(
    train_dataset,
    epochs=15,
    validation_data=val_dataset,
    callbacks=callbacks
)

test_mae = model.evaluate(test_dataset)[1]
print(f"Test MAE: {test_mae:.2f}")

plt.figure()
plt.plot(history.history["mae"], "bo", label="Training MAE")
plt.plot(history.history["val_mae"], "b", label="Validation MAE")
plt.title("Training and Validation MAE (Deeper LSTM)")
plt.legend()
plt.show()

