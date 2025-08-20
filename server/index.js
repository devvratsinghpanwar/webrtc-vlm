const express = require("express");
const http = require("http");
const { Server } = require("socket.io");

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: [
      "*",
      "http://localhost:3000",
      "https://50d54ab3d47b3dd9f74904f54d39cc4a.serveo.net",
    ],
    methods: ["GET", "POST"],
  },
});

const PORT = 4000;
const ROOM_ID = "main-room";

app.get("/", (req, res) => {
  res.send("WebRTC Signaling Server");
});

io.on("connection", (socket) => {
  console.log(`User connected: ${socket.id}`);

  socket.on("join-room", () => {
    socket.join(ROOM_ID);
    console.log(`User ${socket.id} joined ${ROOM_ID}`);

    const clients = io.sockets.adapter.rooms.get(ROOM_ID);
    const numClients = clients ? clients.size : 0;
    console.log(`Room has ${numClients} client(s)`);

    if (numClients === 2) {
      for (const clientId of clients) {
        if (clientId !== socket.id) {
          io.to(clientId).emit("peer-joined");
        }
      }
    }
  });

  socket.on("phone-ready", () => {
    console.log(`Phone ${socket.id} is ready.`);
    const clients = io.sockets.adapter.rooms.get(ROOM_ID);
    if (clients) {
      for (const id of clients) {
        if (id !== socket.id) {
          io.to(id).emit("start-offer");
        }
      }
    }
  });

  // Relay mode selection from phone to other peer(s) using 'mode-select'
  socket.on("mode-select", (mode) => {
    const clients = io.sockets.adapter.rooms.get(ROOM_ID);
    if (clients) {
      for (const id of clients) {
        if (id !== socket.id) {
          io.to(id).emit("mode-select", mode);
          console.log(`Relaying mode-select '${mode}' from ${socket.id} to ${id}`);
        }
      }
    }
  });

  socket.on("offer", (offer) => {
    socket.to(ROOM_ID).emit("offer", offer);
    console.log(`Relaying offer from ${socket.id}`);
  });

  socket.on("answer", (answer) => {
    socket.to(ROOM_ID).emit("answer", answer);
    console.log(`Relaying answer from ${socket.id}`);
  });

  socket.on("ice-candidate", (candidate) => {
    socket.to(ROOM_ID).emit("ice-candidate", candidate);
  });

  socket.on("disconnect", () => {
    console.log(`User disconnected: ${socket.id}`);
  });
});

server.listen(PORT, () => {
  console.log(`Signaling server listening at :${PORT}`);
});


