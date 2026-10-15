# Development CPU mock. No weights, package installer or external APIs.
# Digest observed from Docker Hub during the actual Linux mock build on 2026-10-14.
FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d
WORKDIR /app
COPY hala /app/hala
COPY tests /app/tests
RUN python -m unittest discover -s tests -p 'test_serving.py'
USER 65532:65532
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
CMD ["python", "-m", "hala.serving"]
