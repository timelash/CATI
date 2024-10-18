import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import os

# Loads the CIFAR-100 dataset. This dataset consists of 100 classes of small images (32x32 pixels)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data(label_mode="fine")

# CIFAR-100 class names (fine labels)
class_names_cifar100 = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy",
    "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", 
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest",
    "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", 
    "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid", 
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", 
    "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", 
    "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", 
    "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", 
    "woman", "worm"
]

# We're only interested in 'bicycle' and 'train'
class_names = ["bicycle", "train"]
bicycle_class_idx = class_names_cifar100.index("bicycle")
train_class_idx = class_names_cifar100.index("train")

# Filter training data to keep only 'bicycle' and 'train' images
train_filter = (train_labels == bicycle_class_idx) | (train_labels == train_class_idx)
test_filter = (test_labels == bicycle_class_idx) | (test_labels == train_class_idx)

train_images = train_images[train_filter.flatten()]
train_labels = train_labels[train_filter.flatten()]
test_images = test_images[test_filter.flatten()]
test_labels = test_labels[test_filter.flatten()]

# The labels are converted into binary form: 0 for "bicycle" and 1 for "train"
# This simplifies the classification to a binary problem
train_labels = (train_labels == train_class_idx).astype(int)
test_labels = (test_labels == train_class_idx).astype(int)

# Normalize the images
# Pixel values are scaled from the range [0, 255] to the range [0, 1] by dividing by 255
# This helps the model train more efficiently by ensuring that the input values are small
train_images, test_images = train_images / 255.0, test_images / 255.0

# The file path where the model is saved. If the model exists, it is loaded from disk
model_path = "cifar100_bicycle_train_model.h5"
if os.path.exists(model_path):
    # Load the model if it exists
    model = tf.keras.models.load_model(model_path)
    print("Model loaded from disk.")
else:
    # Build and train the model if it doesn't exist
    # Conv2D layers: Apply convolution filters to the input image (used for detecting patterns like edges)
    # MaxPooling2D layers: Reduce the dimensionality of the data by taking the maximum value over a region (used for down-sampling)
    # Flatten layer: Converts the 2D image output to a 1D vector
    # Dense layers: Fully connected layers for binary classification
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(1),  # Binary classification for 'bicycle' and 'train'
        ]
    )
    
    # Configures the model for training 
    # Adam optimizer: An efficient gradient descent optimization algorithm
    # BinaryCrossentropy loss: A loss function for binary classification
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(
        train_images,
        train_labels,
        epochs=10,
        validation_data=(test_images, test_labels),
    )

    # Save the model
    model.save(model_path)
    print("Model saved to disk.")

# Classify the image
# tf.expand_dims(): Adds a batch dimension to the input image (since the model expects a batch of images)
# model.predict(): Uses the trained model to predict the class of the image
# tf.sigmoid(): Converts the raw prediction (logit) into a probability
# tf.round(): Rounds the output to 0 or 1 (binary classification)
# Returns: The predicted class name ("bicycle" or "train")
def classify_image(image):
    img_array = tf.expand_dims(image, 0)  # Create a batch
    predictions = model.predict(img_array)
    predicted_class = int(tf.round(tf.sigmoid(predictions[0])).numpy())
    return class_names[predicted_class]


# Display the image with its predicted class
def show_image_with_prediction(image, true_label):
    true_label = int(true_label)
    predicted_label = classify_image(image)
    plt.figure()
    plt.imshow(image)
    plt.title(f"Predicted: {predicted_label}, True: {class_names[true_label]}")
    plt.axis("off")
    plt.show()


class_pred = classify_image(test_images[2])
show_image_with_prediction(test_images[2], test_labels[2])

