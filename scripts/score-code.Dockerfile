FROM python:3.11-slim
RUN pip install --no-cache-dir numpy==1.26.4
WORKDIR /scorer
COPY scripts/score_code.py scripts/code_answers.py ./
ENTRYPOINT ["python", "score_code.py"]
