# %%
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.preprocessing import image

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# %% [markdown]
# ## Preprocessing

# %%
TESTING = False
NUM_TRAINING_IMGS = 2000
# NUM_TESTING_IMGS = 10

test_image_dir = ""
hpc_image_dir = "/groups/CS156b/data/"

image_dir = test_image_dir if TESTING else hpc_image_dir

csv_path = (
    test_image_dir + "train/train2023.csv"
    if TESTING
    else "/groups/CS156b/data/student_labels/train2023.csv"
)
csv_pd = pd.read_csv(csv_path)
print(csv_pd.shape)
csv_pd = csv_pd.head(NUM_TRAINING_IMGS)

if TESTING:
    # List of strings to search for
    search_strings = [
        "pid00002",
        "pid00003",
        "pid00004",
        "pid00005",
        "pid00006",
        "pid00007",
        "pid00008",
        "pid00009",
        "pid000010",
        "pid000011",
        "pid000012",
        "pid000013",
        "pid000014",
        "pid000015",
        "pid000016",
    ]

    # Create a regular expression pattern to match any of the strings
    pattern = "|".join(search_strings)

    # Filter rows where 'Path' column contains any of the search strings
    csv_pd = csv_pd[csv_pd["Path"].str.contains(pattern)]
    csv_pd.head(5)

# csv_pd = csv_pd.head(NUM_TRAINING_IMGS)


# %%
### Preprocessing the Y data
label_names = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Pneumonia",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

transforms = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0))]
)


processor = ColumnTransformer(
    transformers=[
        ("num", transforms, label_names),
    ]
)

Y_train = csv_pd[label_names]
Y_train = processor.fit_transform(Y_train)


# Preprocessing the X data
image_size = (224, 224)

train_image = []
for i, path in enumerate(csv_pd["Path"]):
    img = image.load_img(image_dir + path, target_size=(400, 400, 3))
    img = image.img_to_array(img)
    img = img / 255
    train_image.append(img)
X_train = np.array(train_image)

# Splitting the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=42
)

# %%
model = Sequential()
model.add(
    Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(400, 400, 3))
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())  # Batch normalization layer
model.add(Dropout(0.25))

model.add(Conv2D(filters=32, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())  # Batch normalization layer
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())  # Batch normalization layer
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())  # Batch normalization layer
model.add(Dropout(0.25))

# Flatten layer
model.add(Flatten())

# Dense layers
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())  # Batch normalization layer
model.add(Dropout(0.5))

model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())  # Batch normalization layer
model.add(Dropout(0.5))

# Output layer
model.add(Dense(9, activation="sigmoid"))

# %%
model.summary()


# %%
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# %%
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)

# %%
model.save("model2.keras")