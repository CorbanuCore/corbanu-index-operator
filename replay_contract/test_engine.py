import sys
import tempfile
import time
import unittest
from pathlib import Path
from . import engine as e

HEALTH_SERVER='''
import json,sys
from http.server import BaseHTTPRequestHandler,HTTPServer
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200);self.end_headers();self.wfile.write(b'{"status":"ok"}')
HTTPServer(('127.0.0.1',int(sys.argv[1])),Handler).serve_forever()
'''


class ProcessLifecycleTests(unittest.TestCase):
    def test_fixed_launch_controls(self):
        args=e.arguments('/bin/llama-server','/models/first.gguf',12345)
        for key,value in [('--parallel','1'),('--cache-ram','0'),('--moe-stream-io-threads','1'),
                          ('--spec-type','none'),('-c','65536')]:
            self.assertEqual(args[args.index(key)+1],value)
        self.assertIn('--no-context-shift',args);self.assertIn('--no-warmup',args)

    def test_real_child_is_reaped_on_success_and_exception(self):
        with tempfile.TemporaryDirectory() as root:
            for failure in (False,True):
                port=e.free_port();command=[sys.executable,'-c',HEALTH_SERVER,str(port)]
                try:
                    with e.server(command,'http://127.0.0.1:'+str(port),Path(root)/str(failure),5) as (process,remaining):
                        self.assertIsNone(process.poll());self.assertGreater(remaining,0)
                        if failure:raise ValueError('synthetic caller failure')
                except ValueError:
                    self.assertTrue(failure)
                self.assertIsNotNone(process.poll())

    def test_deadline_kills_child_without_waiting_for_caller(self):
        with tempfile.TemporaryDirectory() as root:
            port=e.free_port();command=[sys.executable,'-c',HEALTH_SERVER,str(port)]
            with self.assertRaises(TimeoutError):
                with e.server(command,'http://127.0.0.1:'+str(port),Path(root)/'expiry.log',1) as (process,remaining):
                    time.sleep(1.2)
                    self.assertIsNotNone(process.poll())
