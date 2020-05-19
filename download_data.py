import pickle
import os.path
import io
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive']


def download_from_drive(data_dir="data/"):
    """
    Downloads files from shared data folder to local folder.

    # Arguments
        data_dir: Specifies relative directory to save data to. Default "data/"
    """
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    # Initialise service
    service = build('drive', 'v3', credentials=creds)

    page_token = None
    file_list = []
    while True:
        response = service.files().list(q="'19szE0nEAeR6W3wRTNoC1HX6mJUJg-m8t' in parents",
                                        spaces='drive',
                                        fields='nextPageToken, files(id, name)',
                                        pageToken=page_token).execute()
        for file in response.get('files', []):
            # Process change
            file_list.append({"id": file.get('id'), "name": file.get('name')})
            print('Found file: %s (%s)' % (file.get('name'), file.get('id')))
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break

    # Create data directory if it doesn't exist
    if not os.path.isdir(data_dir):
        print("Data folder not found. Creating...\n")
        os.makedirs(data_dir)

    for file in file_list:

        print("\rDownloading {}: 0%".format(
            file['name']), end="")

        # Download file
        request = service.files().get_media(fileId=file['id'])
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            print("\rDownloading {}: {}%".format(
                file['name'], int(status.progress() * 100)), end="")

        print("")

        # Write byte stream to file
        if not os.path.exists('{}/{}'.format(data_dir, file['name'])):
            with open('{}/{}'.format(data_dir, file['name']), 'w'):
                pass

        with open('{}/{}'.format(data_dir, file['name']), 'w+b') as f:
            f.write(fh.getvalue())

        print("{} saved to data folder.\n".format(file['name']))


def unzip_data(data_dir="data/", save_dir="data/"):
    """
    Unzips files saved in local data folder and organises to subdirectories.
    """

    # Get list of all zip files in data directory
    files_to_unzip = glob.glob("data/*.zip")

    # TODO: Check if files are unzipped

    print("Unzipping: {}".format(files_to_unzip))

    # Need to hardcode unzipping method since each dataset is different
    if 'data\\amazonreviews.zip' in files_to_unzip:
        file_index = files_to_unzip.index('data\\amazonreviews.zip')
        file = files_to_unzip[file_index]
        files_to_unzip.remove(file)

        if not os.path.exists("data/amazon-reviews"):
            os.mkdir("data/amazon-reviews")

        with zipfile.ZipFile(file) as z:
            z.extractall("data/temp")

        with bz2.BZ2File("data/temp/train.ft.txt.bz2") as train_file:
            train_data = open("data/amazon-reviews/train.txt", "wb+")

            train_data.write(train_file.read())

        with bz2.BZ2File("data/temp/test.ft.txt.bz2") as txt_file:
            test_data = open("data/amazon-reviews/test.txt", "wb")

            test_data.write(txt_file.read())
        
        shutil.rmtree("data/temp", ignore_errors=True)

    if 'data\\consumer-reviews-of-amazon-products.zip' in files_to_unzip:
        file_index = files_to_unzip.index('data\\consumer-reviews-of-amazon-products.zip')
        zipped = files_to_unzip[file_index]
        files_to_unzip.remove(zipped)

        with zipfile.ZipFile(zipped) as z:
            z.extractall("data/amazon-consumer-reviews")

    if 'data\\coronavirus-covid19-tweets-early-april.zip' in files_to_unzip:
        file_index = files_to_unzip.index('data\\coronavirus-covid19-tweets-early-april.zip')
        zipped = files_to_unzip[file_index]
        files_to_unzip.remove(zipped)

        with zipfile.ZipFile(zipped) as z:
            z.extractall("data/covid19-tweets")

    if 'data\\coronavirus-covid19-tweets-late-april.zip' in files_to_unzip:
        file_index = files_to_unzip.index('data\\coronavirus-covid19-tweets-late-april.zip')
        zipped = files_to_unzip[file_index]
        files_to_unzip.remove(zipped)

        with zipfile.ZipFile(zipped) as z:
            z.extractall("data/covid19-tweets")

    if 'data\\coronavirus-covid19-tweets.zip' in files_to_unzip:
        file_index = files_to_unzip.index('data\\coronavirus-covid19-tweets.zip')
        zipped = files_to_unzip[file_index]
        files_to_unzip.remove(zipped)

        with zipfile.ZipFile(zipped) as z:
            z.extractall("data/covid19-tweets")

    if 'data\\sentiment140.zip' in files_to_unzip:
        file_index = files_to_unzip.index('data\\sentiment140.zip')
        zipped = files_to_unzip[file_index]
        files_to_unzip.remove(zipped)

        with zipfile.ZipFile(zipped) as z:
            z.extractall("data/sentiment-140")

    print("All data extracted")
