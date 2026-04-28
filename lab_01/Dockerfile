# Use the official Python 3.11 slim image as the starting point.
# "slim" = smaller, faster to download than the full image.
FROM python:3.11-slim

# Set the working directory inside the container.
# Anything we copy in or run will happen here.
WORKDIR /app

# Copy requirements first (Docker caches this layer, so installs are skipped
# on rebuilds when requirements.txt hasn't changed).
COPY requirements.txt .

# Install the Python packages listed in requirements.txt.
# --no-cache-dir keeps the image smaller.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container.
COPY . .

# Expose port 8888 so we can reach Jupyter from our browser.
EXPOSE 8888

# Default command when the container starts: run Jupyter Lab,
# listening on all interfaces, no token (fine for local dev).
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
