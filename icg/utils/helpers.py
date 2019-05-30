import wget
from pathlib import Path
import os
import shutil
import zipfile

def create_folder(path):
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except:
        print("An error occured")

def delete_folder(path: str):
    try:
        if os.path.exists(path):
            shutil.rmtree(path)
    except:
        print("An error occured")

def download(url, dest="."):
    try:
        print("Downloading : {} in {}".format(url, dest))
        print("It could take a few minutes")
        file = wget.download(url, out=dest)
        return True
    except:
        print("An error occured : download")

def unzip(file, dest="."):
    try:
        print("Unzip... {} in {} ".format(file, dest))
        zipfilePath = (file)
        zip = zipfile.ZipFile(zipfilePath)
        zip.extractall(dest)
        zip.close()
    except:
        print("An error occured: unzip")

