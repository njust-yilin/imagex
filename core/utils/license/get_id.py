from hashlib import sha256
import os
import uuid

#1 sudo dmidecode -t system | grep Serial
#2 sudo dmidecode -t 2 | grep Serial
#3 sudo dmidecode -t baseboard | grep Serial
#4 sudo dmidecode -t chassis | grep Serial
#5 sudo dmidecode -t system | grep UUID

def get_hardware_id():
    machine_id_path = '/etc/.machine_id'
    if not os.path.exists(machine_id_path):
        os.system(f"sudo sh -c 'echo {uuid.uuid4()} > {machine_id_path}'")
    with open('/etc/.machine_id', 'rb') as fd:
        machine_id = fd.read()

    hardware_info = os.popen('sudo /usr/sbin/dmidecode -t system | grep Serial').readline().lstrip()
    hardware_info += os.popen('sudo /usr/sbin/dmidecode -t system | grep UUID').readline().lstrip()
    hardware_info += os.popen('sudo /usr/sbin/dmidecode -t 2 | grep Serial').readline().lstrip()
    hardware_info += os.popen('sudo /usr/sbin/dmidecode -t baseboard | grep Serial').readline().lstrip()
    hardware_info += os.popen('sudo /usr/sbin/dmidecode -t chassis | grep Serial').readline().strip()
    hardware_info = hardware_info.encode()

    with open('/usr/sbin/dmidecode', 'rb') as fd:
        dmidecode_sign = fd.read()

    hardware_id = sha256(machine_id + hardware_info + dmidecode_sign).hexdigest()
    return hardware_id


if __name__ == "__main__":
    print(get_hardware_id())
