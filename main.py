import os
import time
import multiprocessing

"""
    function for start Flask server
"""
def run_flask_server():
    os.system('./start.sh')

"""
    function for start Streamlit server
"""
def run_streamlit_server():
    time.sleep(4)
    os.system('./run.sh')


if __name__ == '__main__':
    """
        Run Backend Server & Frontend Server at the same time.
    """
    p1 = multiprocessing.Process(target=run_flask_server)
    p2 = multiprocessing.Process(target=run_streamlit_server)

    p1.start()
    p2.start()

    p1.join()
    p2.join()