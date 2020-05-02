from __future__ import print_function
import pickle
import os.path
import io
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from apiclient.http import MediaIoBaseDownload

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive']


def main():
    """Shows basic usage of the Drive v3 API.
    Prints the names and ids of the first 10 files the user has access to.
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
    if not os.path.isdir('data/'):
        print("Data folder not found. Creating...\n")
        os.makedirs('data/')

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
        if not os.path.exists('data/{}'.format(file['name'])):
            with open('data/{}'.format(file['name']), 'w'):
                pass

        with open('data/{}'.format(file['name']), 'w+b') as f:
            f.write(fh.getvalue())

        print("{} saved to data folder.\n")


if __name__ == '__main__':
    main()
