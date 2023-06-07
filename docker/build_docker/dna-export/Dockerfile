FROM python:3.10-slim-bullseye
LABEL name='dna-export'
LABEL email='kwlee@etri.re.kr'

WORKDIR	/dna.export

COPY dna.export-2.1.0-py3-none-any.whl dna.export-2.1.0-py3-none-any.whl
RUN pip install --upgrade pip && \
    pip install dna.export-2.1.0-py3-none-any.whl && \
    rm dna.export-2.1.0-py3-none-any.whl

#ENTRYPOINT [ "dna_export_topics", "--topic" "track-events", "track-motions", "track-features", "-f" ]