# Build stage
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim
WORKDIR /app

# Create non-root user
RUN groupadd -r slicing && useradd -r -g slicing slicing

# Copy dependencies from builder
COPY --from=builder /root/.local /home/slicing/.local

# Copy application code
COPY src /app/src
COPY setup.py /app/

# Environment variables
ENV PATH=/home/slicing/.local/bin:$PATH
ENV PYTHONPATH=/app/src

# Set ownership
RUN chown -R slicing:slicing /app

# Switch to non-root user
USER slicing

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
