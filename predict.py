from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import pandas as pd

TESTING = False
# NUM_TESTING_IMGS = 10

test_image_dir = ""
hpc_image_dir = "/groups/CS156b/data/"

image_dir = test_image_dir if TESTING else hpc_image_dir
test_csv_path = "/groups/CS156b/data/student_labels/test_ids.csv"

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

model = load_model("model1.keras")

if not TESTING:
    test_csv_pd = pd.read_csv(test_csv_path)
    print(test_csv_pd.shape)
    test_img_paths = test_csv_pd["Path"]
    test_img_ids = test_csv_pd["Id"]
else:
    test_img_ids, test_img_paths = enumerate(csv_pd["Path"])    

imgs = []
for img_path in test_img_paths:
    img = image.load_img(image_dir + img_path, target_size=(400, 400))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize pixel values
    imgs.append(img)

# Make predictions using your model
predictions = []
for img in imgs:
    pred = model.predict(img)
    predictions.append(pred.flatten())

# Convert predictions to DataFrame
predictions_df = pd.DataFrame(predictions, columns=label_names)
predictions_df.insert(0, "ID", test_img_ids)

# Write predictions to a CSV file
predictions_df.to_csv("predictions.csv", index=False)