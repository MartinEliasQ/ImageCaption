from pickle import load
from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras import backend as K


def extract_feactures(directory):
    model = VGG19(weights='imagenet')
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = dict()
    for name in listdir(directory):
        filename = "{}/{}".format(directory, name)
        image = load_img(filename, target_size=(224, 224))
        image = image.reshape(
            (1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        image_id = name.split('.')[0]
        features[image_id] = feature
        print('{}'.format(name))
    return features
# load photo features


def load_photo_features(filename, dataset):
        # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features
