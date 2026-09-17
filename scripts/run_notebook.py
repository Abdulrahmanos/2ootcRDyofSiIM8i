"""Execute the project notebook and persist outputs using a local Jupyter kernel."""

import json
import queue
import time
from pathlib import Path

from jupyter_client import KernelManager


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "01_monreader_baseline.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    manager = KernelManager(kernel_name="python3")
    manager.start_kernel(cwd=str(root))
    client = manager.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=60)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            print(f"Executing cell {index + 1}/{len(notebook['cells'])}", flush=True)
            source = cell["source"]
            msg_id = client.execute("".join(source) if isinstance(source, list) else source)
            deadline = time.monotonic() + 900
            error = None
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Cell {index + 1} exceeded 15 minutes")
                try:
                    message = client.get_iopub_msg(timeout=1)
                except queue.Empty:
                    continue
                if message["parent_header"].get("msg_id") != msg_id:
                    continue
                kind, content = message["msg_type"], message["content"]
                if kind == "execute_input":
                    cell["execution_count"] = content["execution_count"]
                elif kind == "stream":
                    cell["outputs"].append({"output_type": kind, "name": content["name"], "text": content["text"]})
                    print(content["text"], end="", flush=True)
                elif kind in {"display_data", "execute_result"}:
                    output = {"output_type": kind, "data": content["data"], "metadata": content["metadata"]}
                    if kind == "execute_result":
                        output["execution_count"] = content["execution_count"]
                    cell["outputs"].append(output)
                elif kind == "error":
                    error = f"{content['ename']}: {content['evalue']}"
                    cell["outputs"].append({"output_type": "error", **{key: content[key] for key in ("ename", "evalue", "traceback")}})
                elif kind == "status" and content["execution_state"] == "idle":
                    break
            path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
            if error:
                raise RuntimeError(f"Cell {index + 1}: {error}")
        print("Notebook completed; all outputs saved.")
    finally:
        path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
        client.stop_channels()
        manager.shutdown_kernel(now=True)


if __name__ == "__main__":
    main()
