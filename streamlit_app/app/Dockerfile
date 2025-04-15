# Streamlit Dockerfile
FROM python:3.10-slim

# Set environment variables to ensure Python runs optimally
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install dependencies from app_requirements.txt
COPY app_requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r app_requirements.txt

# Copy application files
COPY . /app/

# Expose the necessary port
EXPOSE 8080

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]