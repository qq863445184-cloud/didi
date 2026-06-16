from scripts.chapter8_real_file_matrix import run_real_file_matrix


def test_chapter8_real_file_matrix_imports_retrieves_and_answers(tmp_path):
    report = run_real_file_matrix(workspace=tmp_path)

    assert {row["suffix"] for row in report["files"]} == {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".png",
        ".mp3",
        ".wav",
    }
    for row in report["files"]:
        assert row["imported"] is True
        assert row["chunk_count"] >= 1
        assert row["marker"] in row["search"]
        assert row["marker"] in row["answer"]
