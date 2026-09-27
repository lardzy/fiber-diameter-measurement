"""Local preview server with byte ranges for native video seeking."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1] / "out"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.remaining = None
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_head(self):
        path = Path(self.translate_path(self.path))
        self.remaining = None
        if not path.is_file():
            return super().send_head()
        stream = path.open("rb")
        size = path.stat().st_size
        start, end = 0, size - 1
        request_range = self.headers.get("Range")
        if request_range:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", request_range.strip())
            if match:
                lo, hi = match.groups()
                if lo:
                    start = int(lo)
                    end = min(int(hi), end) if hi else end
                elif hi:
                    start = max(0, size - int(hi))
                if start <= end and start < size:
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.remaining = end - start + 1
                    stream.seek(start)
                else:
                    stream.close()
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return None
            else:
                stream.close()
                self.send_error(400, "Unsupported byte range")
                return None
        else:
            self.send_response(200)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Last-Modified", self.date_time_string(path.stat().st_mtime))
        self.end_headers()
        return stream

    def copyfile(self, source, outputfile):
        try:
            if self.remaining is None:
                return super().copyfile(source, outputfile)
            remaining = self.remaining
            while remaining:
                data = source.read(min(1024 * 1024, remaining))
                if not data:
                    break
                outputfile.write(data)
                remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError):
            pass  # A player can abandon a buffered range after a seek.


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 3235), Handler)
    print("Tutorial player: http://127.0.0.1:3235/v2/教程目录.html", flush=True)
    server.serve_forever()
