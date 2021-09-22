import os
import time
import multiprocessing


def run_flask_server():
    os.system('./start.sh')


def run_streamlit_server():
    time.sleep(4)
    os.system('./run.sh')


if __name__ == '__main__':
    p1 = multiprocessing.Process(target=run_flask_server)
    p2 = multiprocessing.Process(target=run_streamlit_server)

    p1.start()
    p2.start()

    p1.join()
    p2.join()