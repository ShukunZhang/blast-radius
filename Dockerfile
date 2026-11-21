# Score image. The coverage map is built from sqlglot eb11aec and committed;
# the build only fetches that commit so the selector can read the original
# sources when it applies a patch.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ADD https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz /tmp/sqlglot.tar.gz
RUN tar -xzf /tmp/sqlglot.tar.gz -C /opt \
    && mv /opt/sqlglot-eb11aec4be8cb2122afe713a4816662c3f0fb2fa /opt/sqlglot \
    && rm /tmp/sqlglot.tar.gz \
    && chmod -R a+rX /opt/sqlglot

WORKDIR /app
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir 'duckdb==2.0.0' 'pandas==3.1.1'
COPY selector.py analyze.py diffutil.py edits.py pack.py probe_small.py uncovered.py helper_edits.py class_users.py class_reads.py startup_cache.py /app/
COPY maps /app/maps

ENV SQLGLOT_ROOT=/opt/sqlglot \
    COVERAGE_MAP=/app/maps/coverage.bin \
    READS_MAP=/app/maps/reads.json \
    CLASSES_MAP=/app/maps/classes.json \
    STARTUP_CACHE=/app/startup.pickle \
    SKIP_INTEGRATION=0 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -B /app/selector.py --prepare && chmod -R a+rX /app

CMD ["python", "-B", "/app/selector.py"]
