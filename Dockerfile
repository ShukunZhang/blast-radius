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
COPY selector.py analyze.py diffutil.py edits.py pack.py /app/
COPY maps /app/maps
RUN chmod -R a+rX /app

ENV SQLGLOT_ROOT=/opt/sqlglot \
    COVERAGE_MAP=/app/maps/coverage.bin \
    PYTHONDONTWRITEBYTECODE=1

CMD ["python", "-B", "/app/selector.py"]
