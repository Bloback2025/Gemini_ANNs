# sample_payload.py
from pathlib import Path
import json

print("[Payload] Executing sample workload...")

# Write the expected output artifact
output_data = {
    "status": "success",
    "metrics": {"benchmark_score": 0.9876}
}

output_path = Path("payload_output.json")
output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
print(f"[Payload] Artifact written to: {output_path.resolve()}")