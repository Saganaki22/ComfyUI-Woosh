import os

import folder_paths


DEFAULT_WOOSH_FOLDER = os.path.join(folder_paths.models_dir, "woosh")
DEFAULT_MMAUDIO_FOLDER = os.path.join(folder_paths.models_dir, "mmaudio")
HIDDEN_WOOSH_FOLDERS = {"TextConditionerA", "TextConditionerV", "hf_cache"}


def _dedupe_paths(paths):
    seen = set()
    result = []
    for path in paths:
        if not path:
            continue
        full = os.path.abspath(os.path.expanduser(path))
        key = os.path.normcase(full)
        if key not in seen:
            seen.add(key)
            result.append(full)
    return result


def get_model_folder_paths(folder_name, default_folder):
    try:
        paths = folder_paths.get_folder_paths(folder_name)
    except Exception:
        paths = []

    if not paths:
        paths = [default_folder]

    return _dedupe_paths(paths)


def get_woosh_folders():
    return get_model_folder_paths("woosh", DEFAULT_WOOSH_FOLDER)


def get_mmaudio_folders():
    return get_model_folder_paths("mmaudio", DEFAULT_MMAUDIO_FOLDER)


def _safe_relative_name(name):
    name = str(name).replace("\\", os.sep).replace("/", os.sep)
    if os.path.isabs(name):
        return os.path.abspath(name)

    name = os.path.normpath(name)
    if name in ("", ".") or name == ".." or name.startswith(f"..{os.sep}"):
        raise ValueError(f"Invalid Woosh model path: {name}")
    return name


def _has_config(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.yaml"))


def resolve_woosh_path(name, preferred_root=None):
    name = _safe_relative_name(name)
    if os.path.isabs(name):
        return name

    roots = []
    if preferred_root:
        roots.append(preferred_root)
    roots.extend(get_woosh_folders())
    roots = _dedupe_paths(roots)

    for root in roots:
        if os.path.basename(os.path.normpath(root)) == name and _has_config(root):
            return root

        candidate = os.path.join(root, name)
        if os.path.isdir(candidate):
            return candidate

    fallback_root = roots[0] if roots else DEFAULT_WOOSH_FOLDER
    return os.path.join(fallback_root, name)


def get_model_root_for_path(path):
    path = os.path.abspath(path)
    for root in sorted(get_woosh_folders(), key=len, reverse=True):
        try:
            if os.path.commonpath([path, root]) == root:
                if path == root:
                    return os.path.dirname(root)
                return root
        except ValueError:
            continue

    return os.path.dirname(path)


def list_woosh_model_names():
    names = set()

    for root in get_woosh_folders():
        if _has_config(root):
            name = os.path.basename(os.path.normpath(root))
            if name not in HIDDEN_WOOSH_FOLDERS:
                names.add(name)

        if not os.path.isdir(root):
            continue

        for entry in os.listdir(root):
            if entry in HIDDEN_WOOSH_FOLDERS:
                continue

            full = os.path.join(root, entry)
            if _has_config(full):
                names.add(entry)

    return sorted(names)
