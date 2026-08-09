from bigocrpdf.utils.comparison import get_batch_statistics


def test_empty_batch_statistics_have_complete_schema():
    assert get_batch_statistics([]) == {
        "total_files": 0,
        "total_input_size_bytes": 0,
        "total_output_size_bytes": 0,
        "total_input_size_mb": 0.0,
        "total_output_size_mb": 0.0,
        "total_size_change_bytes": 0,
        "total_size_change_mb": 0.0,
        "average_size_change_percent": 0.0,
        "total_pages": 0,
        "total_words": 0,
        "files_larger": 0,
        "files_smaller": 0,
        "files_same_size": 0,
    }
