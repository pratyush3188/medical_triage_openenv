FROM python:3.10-slim

# Create a non-root user
RUN useradd -m -u 1000 user

WORKDIR /app

# Set permissions for the working directory
RUN chown -R user:user /app

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

# Switch to the non-root user
USER user

EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860"]
