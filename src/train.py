import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import bird_loader

def train_model(processed_dir="data/processed", models_dir="models", reports_dir="reports/figures", epochs=25, batch_size=32):
    print("initializing tensorflow training pipeline...")
    
    # check that the processed data exists before running
    train_csv = os.path.join(processed_dir, "train.csv")
    val_csv = os.path.join(processed_dir, "val.csv")
    
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError(f"processed data files not found at {processed_dir}. please run src/data_prep.py first.")
    
    # 1. load the data columns mapped by data_prep.py
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    #2. build datasets using utils loader
    train_ds = bird_loader(train_df["full_path"].values, train_df["label"].values, is_train=True, batch_size=batch_size).get_dataset()
    val_ds = bird_loader(val_df["full_path"].values, val_df["label"].values, is_train=False, batch_size=batch_size).get_dataset()
    
    # 3. create a mobile-first tl architecture
    base_model = tf.keras.applications.MobileNetV3Large(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(200, activation="softmax")
    ])
    
    model.build((None, 224, 224, 3))
    
    # 4. compile and train the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # 5. fit the model and save training history
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, "best_bird_model.keras")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        monitor="val_accuracy", 
        save_best_only=True, 
        mode="max", 
        verbose=1)
    
    # 6. execute model fitting
    print("training started...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[checkpoint_callback])
    
    # 7. render performance plots and save to reports directory
    os.makedirs(reports_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("epochs")
    plt.ylabel("loss score")
    plt.legend()
    plt.title("learning metrics")
    plt.savefig(os.path.join(reports_dir, "training_loss.png"))
    print("metric curve reports successfully exported to reports/figures/training_loss.png")
    
if __name__ == "__main__":
    train_model()