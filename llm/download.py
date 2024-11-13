import zipfile
import io

import requests

if __name__ == "__main__":
    files_url = "https://ideami.com/llm_train"
    print("Downloading files using python")
    response = requests.get(files_url)
    zipfile.ZipFile(io.BytesIO(response.content)).extractall("../pytorch_classes")