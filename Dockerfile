FROM python:3.6

WORKDIR /usr/src/app

COPY . .

RUN pip3 install -r requirements.txt

CMD streamlit run app/app.py --server.port 80
