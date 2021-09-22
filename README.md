# Streamlit with Flask api

Codes are based on this github repo:   
https://github.com/pejmanS21/LungSegmentation_Streamlit
## Description
I use **streamlit** as frontend for application and it's sending requests to backend for **Flask**.
all the predictions and processes will run in backend and results will send to frontend for visualization.

## Usage

In order to get the best performance, you have to run `streamlit` server and `flask` server at the same time.
so you can:

    chmod +x start.sh
    ./start.sh
    chmod +x run.sh
    ./run.sh
or 
    
    chmod +x main.sh
    ./main.sh
    
in terminal, also you can run `main.py` to start the application, as well as mentioned methods.

after that application will be serving on `Local URL` http://localhost:8501
and `Network URL` http://192.168.1.6:8501. flask will serve on port `:5000`




