import numpy as np
from tensorflow import keras
print("here")
from tensorflow.keras import layers
from quaternion_rotation import QuaternionRotation
from data_utils import list_stl_files, load_and_preprocess_stl
from skimage.measure import marching_cubes
import tensorflow as tf
from tensorflow.keras import layers, models
import meshio

def build_generator(input_shape, output_shape):
    model = models.Sequential(
        [
            layers.Input(input_shape),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(1024, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(np.prod(output_shape), activation="sigmoid"),
            layers.Reshape(output_shape),
        ]
    )
    return model

def build_discriminator(input_shape):
    model = models.Sequential(
        [
            layers.Input(input_shape),
            QuaternionRotation(),
            layers.Conv3D(32, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation="relu"),
            layers.BatchNormalization(),
            GaussianDiffusion(),
            layers.Conv3D(64, kernel_size=(3, 3, 3), strides=(2, 2, 2), activation="relu"),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model

def build_gan(generator, discriminator):
    model = models.Sequential([generator, discriminator])
    return model

# Define the input shape and number of classes for your 3D dataset
input_shape = (64, 64, 64, 1)  # (Depth, Height, Width, Channels)
num_classes = 10

directory = "./stls"
file_label_pairs = list_stl_files(directory)

X_train = []
y_train = []

for file_path, label in file_label_pairs:
    preprocessed_stl = load_and_preprocess_stl(file_path, input_shape[:-1])
    X_train.append(preprocessed_stl)

    one_hot_label = one_hot_encode_label(label, label_to_index)
    y_train.append(one_hot_label)

X_train = np.array(X_train)
y_train = np.array(y_train)

generator_input_shape = (y_train.shape[1],)  # Assuming y_train contains the one_hot_labels
discriminator_input_shape = input_shape  # Assuming input_shape is the shape of preprocessed_stl

generator = build_generator(generator_input_shape, discriminator_input_shape)
discriminator = build_discriminator(discriminator_input_shape)

# Train the model
discriminator.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# Set the discriminator to be non-trainable when combined with the generator in the GAN model
discriminator.trainable = False

gan = build_gan(generator, discriminator)
gan.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5),
    loss="binary_crossentropy",
)

def train_gan(epochs, batch_size):
    for epoch in range(epochs):
        for batch in range(len(X_train) // batch_size):
            # Select a random batch of samples and labels
            idx = np.random.randint(0, len(X_train), batch_size)
            real_images = X_train[idx]
            real_labels = y_train[idx]

            # Generate fake images
            noise = np.random.normal(0, 1, (batch_size, generator_input_shape[0]))
            fake_images = generator.predict(noise)

            # Concatenate real and fake images
            images = np.concatenate([real_images, fake_images])
            labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])

            # Train the discriminator
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(images, labels)

            # Train the generator
            noise = np.random.normal(0, 1, (batch_size, generator_input_shape[0]))
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        print(f"Epoch {epoch + 1}, Discriminator loss: {d_loss[0]}, Generator loss: {g_loss}")

# Train the GAN
train_gan(epochs=100, batch_size=32)


text_description = "A simple cube"
text_encoding = encode_text(text_description)  # Replace with your text encoding function
generated_mesh = generator.predict(np.array([text_encoding]))[0]

# Assuming generated_mesh is a voxel grid with values in the range [0, 1]
threshold = 0.5
verts, faces, _, _ = marching_cubes(generated_mesh, level=threshold)

output_filename = "output.stl"
mesh = meshio.Mesh(verts, [("triangle", faces)])
meshio.write(output_filename, mesh)

