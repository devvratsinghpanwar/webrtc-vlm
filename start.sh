#!/bin/bash
# start.sh

# Stop any running containers and remove them to ensure a clean start
docker-compose down

# Build and start the services in detached mode
docker-compose up --build -d

echo "Application is starting..."
echo "Open http://localhost:3000 in your browser."