import subprocess
import os
import sys
import pytest

def test_cli_help():
    """Test that the CLI returns help information."""
    result = subprocess.run([sys.executable, "rag_logic.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "RAG System Management CLI" in result.stdout
    assert "--sync" in result.stdout

def test_cli_sync_invocation(tmp_path, monkeypatch):
    """Test that --sync triggers the sync logic and creates index files."""
    # Create a test document folder in current directory for simpler subprocess test
    # or use absolute paths via environment variables if supported by the script.
    
    # Setup test environment
    me_dir = tmp_path / "me"
    me_dir.mkdir()
    (me_dir / "test.txt").write_text("CLI test content.")
    
    index_dir = tmp_path / "index"
    index_dir.mkdir()

    # Resolve absolute path to the script being tested
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag_logic.py"))
    
    # Run in the tmp_path as CWD so 'me' and 'index' are looked up there
    result = subprocess.run(
        [sys.executable, script_path, "--sync"],
        cwd=tmp_path,
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert "Starting manual index sync" in result.stdout
    assert "Sync complete!" in result.stdout
    assert "1 files processed" in result.stdout
    
    # Verify files were created in the index folder within tmp_path
    assert (tmp_path / "index" / "index.faiss").exists()
    assert (tmp_path / "index" / "metadata.pkl").exists()
