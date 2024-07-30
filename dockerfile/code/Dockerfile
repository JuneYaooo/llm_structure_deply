FROM june666666/llm_structure_env:structure_20240729

WORKDIR /code
COPY ./tmp/code/*.py /code/
COPY ./tmp/code/*.sh /code/
COPY ./tmp/code/src /code/src
COPY ./tmp/code/utils /code/utils
COPY ./tmp/code/log /code/log

RUN apt-get update && rm -rf /var/lib/apt/lists/*
EXPOSE 46000

CMD ["bash", "-c", "source activate llm_structure && python /code/infer_api.py"]
