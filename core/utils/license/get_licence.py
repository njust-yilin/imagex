import rsa
import json
import time
from file_sha256 import get_file_sha256
from hashlib import sha256
import os
import sys

pub_key_pem = b'-----BEGIN RSA PUBLIC KEY-----\nMIGJAoGBAKU0tVAKTliO4jMwXWMHl/XABUUoXmcGRS5S1cL0BwZLM7bnL7D5DsAF\ngCHVW0SnPNjm078uxeQw+nHb0uf0qzUVXv0HJl9lKUniPqGLpJUNHNolREUhCSGe\nHzx568N/wGr2oW1CMlojf1VqAw9WNpCyBhOYkV/oBqZEEkSRv/RBAgMBAAE=\n-----END RSA PUBLIC KEY-----\n'


# 每次加密117 解密128
def rsa_encrypt(content: bytes):
    pub_key = rsa.PublicKey.load_pkcs1(pub_key_pem)
    length = len(content)
    index = 0
    crypto = b''
    while index < length:
        tmp = rsa.encrypt(content[index:index+117], pub_key)
        crypto += tmp
        index += 117
    return crypto



def get_licence(hardware_id, expires=-1):
    if expires >= 0:
        expires = time.time() + 60*60*24*expires

    # verification_license进行签名，防止篡改
    dir = os.path.dirname(__file__)
    dir = os.path.abspath(dir)
    verify_license_sha256 = get_file_sha256(f'{dir}/verification_license.py')
    hardware_id = sha256((hardware_id + verify_license_sha256).encode()).hexdigest()
    license_info = {
        'hardware_id': hardware_id,
        'author': 'yilin',
        'organization': 'imagex',
        'license_version': '1.0',
        'expires': expires
    }
    content = json.dumps(license_info).encode()
    print(content)

    crypto = rsa_encrypt(content)
    with open('LICENSE', 'wb') as fd:
        fd.write(crypto)

    return crypto

if __name__ == '__main__':
    # hardware_id = '7c9bd6a2f5400777e9e7cdf1eb446a16239887de6b8d3b18a830a0b21b9b64f6'
    hardware_id = sys.argv[1]
    license = get_licence(hardware_id)
    # from pathlib import Path
    # os.system(f'mv LICENSE {Path.home()}/imagex_data')