import json

from evaltools.ground_truth_files import (
    cites_source_file,
    knowledge_graph_path,
    load_testdata,
    questions_path,
    remove_source_file,
    write_qa_pairs,
)


def test_paths_are_named_after_source_file(tmp_path):
    assert questions_path(tmp_path, "Testpdf.pdf") == tmp_path / "Testpdf.pdf.jsonl"
    assert knowledge_graph_path(tmp_path, "Testpdf.pdf") == tmp_path / "Testpdf.pdf.kg.json"


def test_write_qa_pairs_overwrites_or_appends(tmp_path):
    path = tmp_path / "nested" / "questions.jsonl"
    write_qa_pairs(path, [{"question": "q1", "truth": "t1"}], append=False)
    write_qa_pairs(path, [{"question": "q2", "truth": "t2"}], append=True)
    assert [row["question"] for row in load_testdata(path)] == ["q1", "q2"]

    write_qa_pairs(path, [{"question": "q3", "truth": "t3"}], append=False)
    assert [row["question"] for row in load_testdata(path)] == ["q3"]


def test_load_testdata_from_folder_reads_only_jsonl_files(tmp_path):
    write_qa_pairs(tmp_path / "b.pdf.jsonl", [{"question": "from b", "truth": ""}], append=False)
    write_qa_pairs(tmp_path / "a.pdf.jsonl", [{"question": "from a", "truth": ""}], append=False)
    (tmp_path / "a.pdf.kg.json").write_text(json.dumps({"nodes": []}))

    assert [row["question"] for row in load_testdata(tmp_path)] == ["from a", "from b"]


def test_cites_source_file_matches_only_that_file():
    assert cites_source_file("Answer [Testpdf.pdf#page=3]", "Testpdf.pdf")
    assert cites_source_file("Answer [Testpdf.pdf]", "Testpdf.pdf")
    assert not cites_source_file("Answer [OtherTestpdf.pdf#page=3]", "Testpdf.pdf")
    assert not cites_source_file("Answer [employee_handbook.pdf#page=1]", "Testpdf.pdf")


def test_remove_source_file_deletes_its_files_and_combined_questions(tmp_path):
    by_file_dir = tmp_path / "ground_truth_by_file"
    questions_file = questions_path(by_file_dir, "Testpdf.pdf")
    graph_file = knowledge_graph_path(by_file_dir, "Testpdf.pdf")
    write_qa_pairs(questions_file, [{"question": "f1", "truth": "[Testpdf.pdf#page=1]"}], append=False)
    graph_file.write_text("{}")
    other_questions_file = questions_path(by_file_dir, "employee_handbook.pdf")
    write_qa_pairs(other_questions_file, [{"question": "hr", "truth": "[employee_handbook.pdf#page=2]"}], append=False)

    combined = tmp_path / "ground_truth.jsonl"
    write_qa_pairs(
        combined,
        [
            {"question": "keep", "truth": "Answer [employee_handbook.pdf#page=2]"},
            {"question": "drop", "truth": "Answer [Testpdf.pdf#page=4]"},
            {"question": "drop too", "truth": "Answer [employee_handbook.pdf#page=1] [Testpdf.pdf#page=9]"},
        ],
        append=False,
    )

    deleted_paths, removed_count = remove_source_file(by_file_dir, combined, "Testpdf.pdf")

    assert set(deleted_paths) == {questions_file, graph_file}
    assert not questions_file.exists()
    assert not graph_file.exists()
    assert other_questions_file.exists()
    assert removed_count == 2
    assert [row["question"] for row in load_testdata(combined)] == ["keep"]


def test_remove_source_file_with_nothing_to_remove(tmp_path):
    combined = tmp_path / "ground_truth.jsonl"
    write_qa_pairs(combined, [{"question": "keep", "truth": "[a.pdf#page=1]"}], append=False)

    deleted_paths, removed_count = remove_source_file(tmp_path / "missing", combined, "Testpdf.pdf")

    assert deleted_paths == []
    assert removed_count == 0
    assert [row["question"] for row in load_testdata(combined)] == ["keep"]
