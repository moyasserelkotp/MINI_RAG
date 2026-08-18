import pytest
import os
from controllers.DataController import DataController
from controllers.ProjectController import ProjectController

def test_filename_path_traversal_sanitization():
    dc = DataController()
    dangerous_filename = "../../../etc/passwd.txt"
    cleaned = dc.get_clean_file_name(dangerous_filename)
    
    # Assert path separators are removed completely
    assert "/" not in cleaned
    assert "\\" not in cleaned
    assert ".." in cleaned or "etc" in cleaned

def test_generate_unique_filepath_contained():
    dc = DataController()
    dangerous_filename = "../../../secret.pdf"
    project_id = "proj-1"
    
    file_path, file_id = dc.generate_unique_filepath(dangerous_filename, project_id)
    project_dir = ProjectController().get_project_path(project_id)
    
    # Ensure generated absolute path is located inside the project directory
    normalized_file = os.path.normpath(file_path)
    normalized_proj = os.path.normpath(project_dir)
    assert normalized_file.startswith(normalized_proj)
