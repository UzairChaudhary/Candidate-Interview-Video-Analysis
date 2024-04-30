from werkzeug.utils import secure_filename
import os
import requests
import tempfile



def download_file(url, download_folder):
    response = requests.get(url)

    # Create the download folder if not exists
    os.makedirs(download_folder, exist_ok=True)

    # Generate a temporary filename
    _, temp_filename = tempfile.mkstemp(suffix='.pdf', dir=download_folder)

    with open(temp_filename, 'wb') as file:
        file.write(response.content)

    print("File Name: ",temp_filename)
    return temp_filename