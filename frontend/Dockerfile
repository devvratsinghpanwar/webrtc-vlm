# frontend/Dockerfile

# --- Stage 1: Build the Next.js application ---
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package.json and lock files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . .

# Copy the wasm files for onnxruntime-web to the public folder
# This ensures they are available in the final build
RUN cp -r node_modules/onnxruntime-web/dist/*.wasm public/

# Build the Next.js app for production
RUN npm run build

# --- Stage 2: Create the final production image ---
FROM node:18-alpine

WORKDIR /app

# Copy the built app from the 'builder' stage
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/public ./public

# Expose the port Next.js runs on
EXPOSE 3000

# The command to start the app
CMD ["npm", "start"]