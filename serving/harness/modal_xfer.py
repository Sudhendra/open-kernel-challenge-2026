"""SSH-free file transfer for RunPod pods, via a Modal web app (CPU only; bills seconds per request).

  XFER_TOKEN=<hex> modal deploy harness/modal_xfer.py      # prints the app URL
  GET  /boot?t=TOKEN            -> lanes/moe_tools/pod_boot.sh
  GET  /bundle?t=TOKEN          -> lane.tgz (baked at deploy from XFER_BUNDLE)
  POST /result?t=TOKEN&name=f   -> stored on volume okc-muse-weights at xfer/<f>
  modal volume get okc-muse-weights xfer/<f> out/
"""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
SERVING = os.path.dirname(HERE)
BUNDLE = os.environ.get("XFER_BUNDLE", "")
TOKEN = os.environ.get("XFER_TOKEN", "")

app = modal.App("okc-xfer")
vol = modal.Volume.from_name("okc-muse-weights", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi[standard]==0.115.12")
if BUNDLE:
    image = image.add_local_file(BUNDLE, "/data/lane.tgz", copy=True)
image = image.add_local_file(os.path.join(SERVING, "lanes", "moe_tools", "pod_boot.sh"), "/data/pod_boot.sh", copy=True).env(
    {"XFER_TOKEN": TOKEN}
)


@app.function(image=image, volumes={"/vol": vol}, timeout=600)
@modal.asgi_app()
def web():
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import FileResponse, PlainTextResponse

    api = FastAPI()
    tok = os.environ["XFER_TOKEN"]

    def check(t: str):
        if not tok or t != tok:
            raise HTTPException(status_code=404)

    @api.get("/boot")
    def boot(t: str = ""):
        check(t)
        return FileResponse("/data/pod_boot.sh", media_type="text/plain")

    @api.get("/bundle")
    def bundle(t: str = ""):
        check(t)
        return FileResponse("/data/lane.tgz", media_type="application/gzip")

    @api.post("/result")
    async def result(request: Request, t: str = "", name: str = ""):
        check(t)
        name = os.path.basename(name) or "unnamed"
        body = await request.body()
        os.makedirs("/vol/xfer", exist_ok=True)
        with open(f"/vol/xfer/{name}", "wb") as f:
            f.write(body)
        vol.commit()
        return PlainTextResponse(f"stored {name} {len(body)} bytes\n")

    return api
