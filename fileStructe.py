import os

# Define the structure
structure = {
    "app": [
        "__init__.py",
        "main.py",
        {"models": ["__init__.py", "candidate_model.py"]},
        {"routes": ["__init__.py", "candidate_routes.py"]},
        {"services": ["__init__.py", "scoring_service.py"]},
        {"utils": ["__init__.py", "preprocessing.py"]},
        "static/",
        "templates/",
    ],
    "tests": ["test_main.py", "test_scoring.py"],
    ".": ["requirements.txt", "README.md"],
}

# Function to create the structure
def create_structure(base_path, structure):
    if isinstance(structure, dict):
        for folder, contents in structure.items():
            folder_path = os.path.join(base_path, folder)
            os.makedirs(folder_path, exist_ok=True)
            create_structure(folder_path, contents)
    elif isinstance(structure, list):
        for item in structure:
            if isinstance(item, dict):
                create_structure(base_path, item)
            else:
                file_path = os.path.join(base_path, item)
                if item.endswith("/"):
                    os.makedirs(file_path, exist_ok=True)
                else:
                    with open(file_path, "w") as f:
                        f.write("")  # Create an empty file

# Run the script
project_name = "ai_candidate_screening"
os.makedirs(project_name, exist_ok=True)
create_structure(project_name, structure)
print(f"Project '{project_name}' structure created!")
