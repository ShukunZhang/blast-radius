# Maps are built from sqlglot eb11aec during the image build. The run only
# reads them, then replays candidate tests against each patch.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ENV PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
    SKIP_INTEGRATION=0 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r /app/requirements.lock

ADD https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz /tmp/sqlglot.tar.gz
RUN tar -xzf /tmp/sqlglot.tar.gz -C /opt \
    && mv /opt/sqlglot-eb11aec4be8cb2122afe713a4816662c3f0fb2fa /sqlglot \
    && rm /tmp/sqlglot.tar.gz

COPY map_tests.py selector.py inline_pool.py observe.py escape_map.py /app/
WORKDIR /sqlglot
RUN PYTHONPATH=/sqlglot python /app/map_tests.py /app/coverage.json \
    && PYTHONPATH=/sqlglot python /app/observe.py /app/observed.json \
    && python -c "import json; d=json.load(open('/app/observed.json')); assert d.get('comparisons') and d.get('escapes')" \
    && python -m compileall -q /sqlglot /app \
    && chmod -R a+rX /sqlglot /app

CMD ["python", "/app/selector.py"]
