import os


IMAGE_SUFFIX = ['.JPG', '.JPEG', '.PNG', '.BMP']

def is_image(filename):
    _, suffix = os.path.splitext(filename)
    return suffix.upper() in IMAGE_SUFFIX


if __name__ == '__main__':
    print(is_image('test'))