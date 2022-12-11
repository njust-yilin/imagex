import time

def get_timestamp_us():
    return int(round(time.time() * 1000))


if __name__ == '__main__':
    print(get_timestamp_us())