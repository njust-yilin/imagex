from hashlib import sha256


def get_file_sha256(filename):
    with open(filename, 'rb') as fd:
        content = fd.read()
        return sha256(content).hexdigest()
