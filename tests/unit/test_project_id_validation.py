import pytest
from pydantic import ValidationError
from models.db_schemes.project import Project

def test_valid_project_ids():
    valid_ids = [
        "abc",
        "project1",
        "project-1",
        "project_test",
        "a" * 64,
        "A-B_C"
    ]
    for pid in valid_ids:
        project = Project(project_id=pid)
        assert project.project_id == pid

def test_invalid_project_ids():
    invalid_ids = [
        "",  # empty
        " ", # space
        "project name", # space
        "project/name", # slash
        "project@name", # at sign
        "a" * 65 # too long
    ]
    for pid in invalid_ids:
        with pytest.raises(ValidationError):
            Project(project_id=pid)
