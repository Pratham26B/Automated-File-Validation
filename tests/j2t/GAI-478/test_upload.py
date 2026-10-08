"""j2t:
    ticket: GAI-478
    case_ids: [GAI-478-AC1-E2]
    ac_hashes: {AC-1: "sha256:93394ee58d1b46e95e71ee49c9e23fbc4ece98362ec3bfd867dafc1f79054b2f"}
    generated_by_run: run_01m4cywxs0kmgekt55rzhfrmv5
    """

from j2t_guard import j2t_expect
import pytest

# Helper to generate dummy CSV content of given size in bytes
def _make_csv(size_bytes: int) -> bytes:
    # Simple repeat of a line to reach size
    line = b"col1,col2\n"
    repeats, remainder = divmod(size_bytes, len(line))
    return line * repeats + line[:remainder]

@pytest.mark.case("GAI-478-AC1-E2")
def test_upload_unsupported_pdf(j2t, j2t_client):
    # given: a PDF file
    content = b"%PDF-1.4\n%..."
    data = {"data_info": "no"}
    files = {"file": ("doc.pdf", content, "application/pdf")}
    with j2t.act():
        response = j2t_client.post("/upload/", data=data, files=files)
    j2t_expect(response.status_code, 400, ac="AC-1")
    j2t_expect(response.json().get("error"), "Unsupported file type.", ac="AC-1")
