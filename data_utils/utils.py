import os
import tarfile
import hashlib


def md5file(fname):
    hash_md5 = hashlib.md5()
    f = open(fname, "rb")
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()


def download(url, target_dir):
    """Download file from url to target_dir, and check md5sum."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    filepath = os.path.join(target_dir, url.split("/")[-1])
    if not os.path.exists(filepath):
        print("Downloading %s ..." % url)
        os.system("wget -c " + url + " -P " + target_dir)
    else:
        print("File exists, skip downloading. (%s)" % filepath)
    return filepath


def unpack(filepath, target_dir, rm_=False):
    """Unpack the file to the target_dir."""
    print("Unpacking %s ..." % filepath)
    if filepath.endswith('.tar'):
        tar = tarfile.open(filepath)
        tar.extractall(target_dir)
        tar.close()
    elif filepath.endswith('.zip'):
        os.system("unzip -q -j %s -d %s" % (filepath, target_dir))
    if rm_:
        os.remove(filepath)
