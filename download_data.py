import os
import urllib.request
import tarfile
from tqdm import tqdm

class download_progressbar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
        
def download_dataset():
    url = "https://caltech.edu"
    output_dir = "data/raw"
    archive_path = os.path.join(output_dir, "CUB_200_2011.tgz")
        
    if os.path.exists(os.path.join(output_dir, "CUB_200_2011")):
        print("Dataset already exists. Skipping download.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print("Downloading dataset...")
    print("please wait...this may take several minutes")
        
    with download_progressbar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=archive_path, reporthook=t.update_to)
            
    print("extracting dataset...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=output_dir)
            
    os.remove(archive_path)
    print("ready for use!")
        
if __name__ == "__main__":
    download_dataset()
    