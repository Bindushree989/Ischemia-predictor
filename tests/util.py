import hashlib


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()
