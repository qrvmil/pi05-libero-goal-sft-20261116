"""Exercise the actual HTTP body slices, retry, short tail, and ordered ETags."""
import hashlib
import importlib.util
import random
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from huggingface_hub import CommitOperationAdd


class ParallelUploadIntegrityTest(unittest.TestCase):
    def test_roundtrip_with_out_of_order_completion_and_short_last_part(self):
        module_path=Path(__file__).with_name('parallel_lfs.py')
        self.assertTrue(module_path.exists(),'parallel HTTP multipart transport is not implemented')
        from parallel_lfs import upload_parts_parallel
        data=random.Random(42).randbytes(5*1024*1024+123)
        part_size=1024*1024
        received={}
        attempts={}
        lock=threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            protocol_version='HTTP/1.1'

            def do_PUT(self):
                index=int(self.path[1:])
                body=self.rfile.read(int(self.headers['Content-Length']))
                with lock:
                    attempts[index]=attempts.get(index,0)+1
                    attempt=attempts[index]
                if index==2 and attempt==1:
                    self.send_response(503);self.send_header('Content-Length','0');self.end_headers();return
                time.sleep(0.05 if index==0 else 0.001)
                with lock:received[index]=body
                self.send_response(200)
                self.send_header('ETag',hashlib.sha256(body).hexdigest())
                self.send_header('Content-Length','0')
                self.end_headers()

            def log_message(self,*args):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'weights.bin';source.write_bytes(data)
            server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
            worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
            try:
                urls=[f'http://127.0.0.1:{server.server_port}/{i}' for i in range(6)]
                result=upload_parts_parallel(CommitOperationAdd(path_in_repo='weights.bin',path_or_fileobj=str(source)),
                                             urls,part_size,workers=4)
                rebuilt=b''.join(received[i] for i in range(6))
                self.assertEqual(hashlib.sha256(rebuilt).digest(),hashlib.sha256(data).digest())
                self.assertEqual(len(received[5]),123)
                self.assertEqual(attempts[2],2)
                for i,headers in enumerate(result):
                    self.assertEqual(headers['etag'],hashlib.sha256(data[i*part_size:(i+1)*part_size]).hexdigest())
            finally:
                server.shutdown();server.server_close();worker.join()


if __name__=='__main__':
    unittest.main()
