from pathlib import Path
import tensorflow as tf

train_dataset = tf.keras.utils.image_dataset_from_directory( 
    "../../data/train",
    image_size=(150, 150),
    batch_size=32,
)

validation_dataset=tf.keras.utils.image_dataset_from_directory(
    "../../data/validation",
    image_size=(150, 150),
    batch_size=32,
)


class_count=len(validation_dataset.class_names)

def activation_layer(name):
    if name == "leaky_relu":
        return tf.keras.layers.LeakyReLU(negative_slope=0.1)
    return tf.keras.layers.Activation(name)


def create_model(hidden_activation, class_count):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(150, 150, 3)),
        tf.keras.layers.Rescaling(1.0 / 255),
        tf.keras.layers.Conv2D(32, 3, padding="same"),
        activation_layer(hidden_activation),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding="same"),
        activation_layer(hidden_activation),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(class_count, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

activation_names = ["relu", "tanh", "leaky_relu"]
models = {}
histories = {}

for activation_name in activation_names:
    tf.keras.utils.set_random_seed(42)
    candidate_model = create_model(activation_name, class_count)
    histories[activation_name] = candidate_model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=50,
        verbose=0,
    )
    models[activation_name] = candidate_model
    experiment_dir = Path(f"experiments/activation-{activation_name}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    candidate_model.save(
        experiment_dir / "model.keras"
    )