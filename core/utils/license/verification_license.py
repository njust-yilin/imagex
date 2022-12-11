import rsa
from hashlib import sha256
import os


private_key_pem = b'-----BEGIN RSA PRIVATE KEY-----\nMIICYAIBAAKBgQClNLVQCk5YjuIzMF1jB5f1wAVFKF5nBkUuUtXC9AcGSzO25y+w\n+Q7ABYAh1VtEpzzY5tO/LsXkMPpx29Ln9Ks1FV79ByZfZSlJ4j6hi6SVDRzaJURF\nIQkhnh88eevDf8Bq9qFtQjJaI39VagMPVjaQsgYTmJFf6AamRBJEkb/0QQIDAQAB\nAoGABdXtzaz/hXtOnDZKJjRfdsvYo8/APe1nxjIg4OkT0nIXmo9iDONPVRMcqpVJ\nywwJRzQoKKmzTdM5FYqJSeG3NBk/kgHhl42/hfhusVU3jvBjEAVZlu5q0VSWH0lO\nfBn5MLuKaTKH4OfPK/3bBBFFjzhhBn2jqRiYMQS/nJ0ykG0CRQDcyU+N8T85BY+v\nd8vFVhkYbmv+L50AeKZQhAf8A1RFMX3AQPxV67HC69NSUFsMe/ifA5f7erHspLqM\nBLNBOc0gOGbsBwI9AL+ODRgaSCzcLA0c5edPnvKhkIlNewFjDdiR2MaoetSZ+j8+\ntWLln4ah1QnLTE/24srp+faogKKLbwCbdwJEPMjjsSV5DX7ddyaZIEQ69oH4E2wS\nYn3U9BfVhul3uvEMOPDrR9BzCUIZ1PCwkHhVE5pOrnqyH3+eqEvm+g8qzTMaCuMC\nPQCLJq3sbGsx6180x2Fbf0OADk3o8BgDEenlAU3wQkO4XYKknvE1Pol8S+NukfiF\nvltR/FZREGchrRid4FECRGiEuRPagZRKEKxkow3mximGpVW1r6oI0MHQ9I0muZ33\nb4h2LG6krJZ3vWnMHKQNY7bJYBn06MErvKLeu4+tVOFcJ8ki\n-----END RSA PRIVATE KEY-----\n'


def get_hardware_id():
    with open('/etc/.machine_id', 'rb') as fd:
        machine_id = fd.read()

    with open('/usr/sbin/dmidecode', 'rb') as fd:
        dmidecode_sign = fd.read()

    hardware_info = os.popen('sudo dmidecode -t system | grep Serial').readline().lstrip()
    hardware_info += os.popen('sudo dmidecode -t system | grep UUID').readline().lstrip()
    hardware_info += os.popen('sudo dmidecode -t 2 | grep Serial').readline().lstrip()
    hardware_info += os.popen('sudo dmidecode -t baseboard | grep Serial').readline().lstrip()
    hardware_info += os.popen('sudo dmidecode -t chassis | grep Serial').readline().strip()
    hardware_info = hardware_info.encode()

    hardware_id = sha256(machine_id + hardware_info + dmidecode_sign).hexdigest()

    return hardware_id

# 每次加密117 解密128

def rsa_decrypt(crypto: bytes):
    private_key = rsa.PrivateKey.load_pkcs1(private_key_pem)
    length = len(crypto)
    index = 0
    content = b''
    while index < length:
        content += rsa.decrypt(crypto[index:index+128], private_key)
        index += 128
    return content.decode()
