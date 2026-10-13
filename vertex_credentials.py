"""Service-account keys for every node in the pack, picked from a dropdown.

Keys live in ComfyUI/user/__vertex_credentials/ (ComfyUI's system-user folder,
from folder_paths.get_system_user_directory):
- outside this node pack, so its repo can never pick them up;
- inside ComfyUI's git-ignored user/ folder;
- blocked from ComfyUI's userdata web API, which refuses '__' folders.

Workflows store only the file name. A connected project_id / service_account
input (e.g. from the Vertex Credentials Loader) overrides the dropdown.
"""

import json
import os

import folder_paths

CREDENTIALS_DIR = folder_paths.get_system_user_directory("vertex_credentials")
NO_FILES = "(no key files found)"
# The dropdown starts here rather than on a real key, so a new node never
# silently bills whichever account happens to sort first.
SELECT = "(select a key)"


def list_credentials():
    """Dropdown entries: the placeholder, then the .json files in
    CREDENTIALS_DIR, read fresh each time ComfyUI asks for node definitions
    (so refreshing the browser picks up a newly added key)."""
    os.makedirs(CREDENTIALS_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(CREDENTIALS_DIR) if f.lower().endswith(".json"))
    return [SELECT, *files] if files else [NO_FILES]


def default_credentials():
    return list_credentials()[0]


CREDENTIALS_TOOLTIP = (f"Service-account key from {CREDENTIALS_DIR}. "
                       "Connected project_id / service_account inputs override it.")


def resolve_credentials(credentials, project_id=None, service_account=None):
    """(project_id, service_account_json) for a request. Connected inputs win;
    whatever they leave empty comes from the selected key file."""
    project_id = (project_id or "").strip()
    service_account = (service_account or "").strip()
    if project_id and service_account:
        return project_id, service_account

    if not credentials or credentials == NO_FILES:
        raise ValueError(f"No service-account key selected. Put the key's .json file in "
                         f"{CREDENTIALS_DIR}, refresh the browser, and pick it in 'credentials' "
                         f"(or connect project_id and service_account).")
    if credentials == SELECT:
        raise ValueError("Pick a service-account key in 'credentials' so the right account is billed "
                         "(or connect project_id and service_account).")
    path = os.path.join(CREDENTIALS_DIR, os.path.basename(credentials))
    if not os.path.isfile(path):
        raise ValueError(f"Key file not found: {path}")

    text = open(path, encoding="utf-8").read().strip()
    try:
        file_project = json.loads(text).get("project_id", "")
    except json.JSONDecodeError as e:
        raise ValueError(f"{credentials} is not valid JSON: {e}")

    project_id = project_id or file_project
    if not project_id:
        raise ValueError(f"{credentials} has no project_id; connect a project_id input.")
    return project_id, service_account or text
