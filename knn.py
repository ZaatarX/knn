import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier

unfiltered_data = pd.read_csv("car.data")
print(unfiltered_data.head())

# transforming non-measurable data into integers

label_encoder = preprocessing.LabelEncoder()

buying = label_encoder.fit_transform(list(unfiltered_data["buying"]))
maint = label_encoder.fit_transform(list(unfiltered_data["maint"]))
door = label_encoder.fit_transform(list(unfiltered_data["door"]))
persons = label_encoder.fit_transform(list(unfiltered_data["persons"]))
lug_boot = label_encoder.fit_transform(list(unfiltered_data["lug_boot"]))
safety = label_encoder.fit_transform(list(unfiltered_data["safety"]))
cls = label_encoder.fit_transform(list(unfiltered_data["class"]))

feature = list(zip(buying, maint, door, persons, lug_boot, safety))
label = list(cls)

feature_train, feature_test, label_train, label_test = sklearn.model_selection.train_test_split(
    feature, label, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)
model.fit(feature_train, label_train)

accuracy = model.score(feature_test, label_test)
print(accuracy)

prediction = model.predict(feature_test)
names = ["unacc", "acc", "good", "vgood"]

for i in range(len(prediction)):
    print("Prediction: ", names[prediction[i]], "\tActual: ", names[label_test[i]], "\tData: ",
          feature_test[i])
