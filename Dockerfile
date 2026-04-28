# Beach labeler — Python only, no apt-get (build works offline / behind strict firewalls).
# Requires no access to deb.debian.org during docker build.
#
# For YouTube frame pulls with ffmpeg + yt-dlp + Node inside the same image, use:
#   docker compose -f docker-compose.full.yml build
# (see Dockerfile.full — that build still needs Debian mirror access.)

FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data

COPY requirements-labeler.txt requirements-train.txt ./
RUN pip install --no-cache-dir -r requirements-labeler.txt -r requirements-train.txt

COPY server.py agreement.py streams_youtube.py pull_youtube_frames.py pull_youtube_clips.py labeling_rules.py ./
COPY training ./training
COPY static ./static

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
