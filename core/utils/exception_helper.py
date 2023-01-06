from loguru import logger

def run_catch_except(func, args=(), kwargs={}):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(e)

if __name__ == '__main__':
    def test(x, y):
        print(x, y)
    run_catch_except(test, kwargs={'x': 1, 'y': 2})
    run_catch_except(lambda x, y: print(x, y), kwargs={'x': 1, 'y': 2})