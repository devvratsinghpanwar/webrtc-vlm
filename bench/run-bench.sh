#!/bin/bash
# bench/run_bench.sh
echo "Simulating a 30s benchmark run..."
# In a real scenario, this would use a tool like 'curl' or a script 
# to hit the API and measure performance.
sleep 30 
echo '{
  "median_latency_ms": 180.5,
  "p95_latency_ms": 250.2,
  "processed_fps": 5.5,
  "uplink_kbps": 0,
  "downlink_kbps": 0
}' > metrics.json
echo "Benchmark finished. Results saved to metrics.json"