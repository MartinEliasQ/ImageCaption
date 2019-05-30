from icg.utils.helpers import (create_folder, delete_folder, download, unzip)
def get(url, dest="/src/data/"):
    create_folder(dest)
    download(url, dest)
    file_name = url.split("/")[-1]
    output_folder = "{}/{}".format(dest,file_name.split(".zip")[0])
    create_folder(output_folder)
    unzip("{}/{}".format(dest,file_name), output_folder)


def get_dataset(url_images, url_text, dest="/src/data/"):
    delete_folder(dest)
    get(url_images, dest)
    get(url_text, dest)    