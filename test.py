from icg.processing.dataset import get_dataset
import sys
import string
sys.path.append("..")

TOKEN_TXT = "../src/txt/Flickr8k.token.txt"
URL_TEXT = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
URL_IMAGES = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
get_dataset(URL_IMAGES, URL_TEXT, dest="./src/dataset")
