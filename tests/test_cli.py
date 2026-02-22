import subprocess
import os
import sys

def test_cli_help():
    """Test that the CLI returns help information."""
    result = subprocess.run([sys.executable, "rag_logic.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "RAG System Management CLI" in result.stdout
    assert "--sync" in result.stdout

def test_cli_sync_invocation(tmp_path):
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
    
    # Run with environment overrides to point the script to temporary test folders
    env = os.environ.copy()
    env["RAG_INDEX_FOLDER"] = str(index_dir)
    env["RAG_SOURCE_FOLDER"] = str(me_dir)
    env["RAG_MOCK_MODEL"] = "true" # Ensure we use mock in subprocess
    
    result = subprocess.run(
        [sys.executable, script_path, "--sync"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env
    )
    
    assert result.returncode == 0
    assert "Starting manual index sync" in result.stdout
    assert "Sync complete!" in result.stdout
    assert "1 files processed" in result.stdout
    
    # Verify files were created in the index folder
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "metadata.pkl").exists()
