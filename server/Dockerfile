# server/Dockerfile

# Use a base image that has both Python and Node.js
FROM nikolaik/python-nodejs:python3.11-nodejs18-bullseye

# Set the working directory
WORKDIR /app

# --- Setup Python Environment ---
# Copy Python requirements file (we will create this next)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- Setup Node.js Environment ---
# Copy Node.js package.json
COPY package.json .

# Install Node.js dependencies
RUN npm install

# --- Copy Application Code ---
# Copy the rest of the server code
COPY . .

# Install Supervisor to manage both processes
RUN apt-get update && apt-get install -y supervisor

# Copy Supervisor configuration file (we will create this next)
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose ports for both servers
EXPOSE 4000
EXPOSE 8000

# The command to run Supervisor, which starts our servers
CMD ["/usr/bin/supervisord"]