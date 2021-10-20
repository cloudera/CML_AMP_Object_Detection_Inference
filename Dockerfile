FROM python:3.6

WORKDIR /usr/src/app

COPY . .

RUN pip3 install -r requirements.txt

# expose default port for streamlit
EXPOSE 8501

CMD streamlit run app/app.py