import pytest
from pydantic import ValidationError
from routes.schemes.data import ProcessRequest, URLIngestRequest

def test_valid_process_request():
    req = ProcessRequest(chunk_size=1000, overlap_size=200, do_reset=1)
    assert req.chunk_size == 1000
    assert req.overlap_size == 200
    assert req.do_reset == 1

def test_invalid_overlap_process_request():
    with pytest.raises(ValidationError) as exc_info:
        ProcessRequest(chunk_size=500, overlap_size=500)
    assert "must be strictly less than chunk_size" in str(exc_info.value)
    
    with pytest.raises(ValidationError):
        ProcessRequest(chunk_size=500, overlap_size=600)

def test_do_reset_legacy_support():
    req1 = ProcessRequest(do_reset=0)
    req2 = ProcessRequest(do_reset=1)
    assert req1.do_reset == 0
    assert req2.do_reset == 1

def test_url_scheme_validation():
    # Valid
    req = URLIngestRequest(url="https://example.com/page")
    assert req.url == "https://example.com/page"
    
    req = URLIngestRequest(url="http://example.com/page")
    assert req.url == "http://example.com/page"

    # Invalid
    with pytest.raises(ValidationError) as exc_info:
        URLIngestRequest(url="ftp://example.com/file")
    assert "Only http:// and https://" in str(exc_info.value)
    
    with pytest.raises(ValidationError):
        URLIngestRequest(url="file:///etc/passwd")
        
    with pytest.raises(ValidationError):
        URLIngestRequest(url="gopher://example.com")
