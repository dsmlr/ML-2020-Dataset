import requests

def download_url(url, save_path, cookies, chunk_size=128):
    r = requests.get(url, stream=True, cookies=cookies)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

if __name__ == '__main__':
    cookies_dict = {"MoodleSession": "cookie_here"}
    url = 'https://onlearn.it.kmitl.ac.th/mod/assign/view.php?id=7311&action=downloadall'

    download_url(url, 'submissions/submission.zip', cookies_dict)
    print("Downloaded submission files.")
