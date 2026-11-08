"""User-authorized curl transport; HTTPX API/snapshot behavior stays unchanged."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import httpx


class CurlTransport(httpx.BaseTransport):
    def __init__(self, executable=None, scratch=None):
        self.executable = executable or shutil.which('curl.exe')
        if not self.executable:
            raise RuntimeError('Existing curl.exe missing; no installation attempted')
        self.scratch = Path(scratch or Path(__file__).resolve().parents[1]/'.runtime/pt0-curl')
        self.scratch.mkdir(parents=True, exist_ok=True)

    def handle_request(self, request):
        url=request.url
        if (request.method != 'GET' or url.scheme != 'https'
            or url.host not in ('zh.wikipedia.org','en.wikipedia.org')
            or url.port not in (None,443) or url.path != '/w/api.php'
            or url.username or url.password or 'authorization' in request.headers):
            raise httpx.ConnectError('Only fixed Wikipedia GET API endpoints permitted',request=request)
        with tempfile.TemporaryDirectory(dir=self.scratch) as directory:
            body,header=Path(directory)/'body.bin',Path(directory)/'headers.bin'
            # Disable implicit .curlrc, never follow redirects, no external
            # links or page commands. Native curl UA + honest API task identity.
            command=[self.executable,'-q','--silent','--show-error','--proto','=https',
              '--max-redirs','0','--connect-timeout','20','--max-time','30',
              '--header','Accept: application/json',
              '--header','Api-User-Agent: StoryAgent-PT0/0.92.0 (personal interactive research)',
              '--output',str(body),'--dump-header',str(header),'--write-out','%{http_code}',
              '--url',str(url)]
            try:
                result=subprocess.run(command,capture_output=True,timeout=35)
            except subprocess.TimeoutExpired as exc:
                raise httpx.ReadTimeout('curl API timeout',request=request) from exc
            if result.returncode:
                raise httpx.ConnectError('curl exit '+str(result.returncode),request=request)
            try:
                status=int(result.stdout.decode('ascii').strip())
                headers={}
                for line in header.read_bytes().decode('iso-8859-1').splitlines():
                    if line.startswith('HTTP/'):
                        headers={}
                    elif ':' in line:
                        k,v=line.split(':',1)
                        if k.lower() not in ('content-length','transfer-encoding','connection'):
                            headers[k.strip()]=v.strip()
                return httpx.Response(status,headers=headers,content=body.read_bytes(),request=request)
            except (ValueError,OSError) as exc:
                raise httpx.RemoteProtocolError('Invalid curl response',request=request) from exc
