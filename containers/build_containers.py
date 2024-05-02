import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from typing import List

import requests

from ViroConstrictor import __prog__

base_path_to_envs = "./ViroConstrictor/workflow/envs/"
base_path_to_scripts = "./ViroConstrictor/workflow/scripts/"
base_path_to_configs = "./ViroConstrictor/workflow/files/"
base_path_to_container_defs = "./containers"
main_upstream_registry = "ghcr.io/rivm-bioinformatics"
main_upstream_api_enpoint = (
    "https://api.github.com/orgs/RIVM-bioinformatics/packages/container/"
)
main_upstream_api_authtoken = os.environ.get("TOKEN")
main_upstream_api_responsetype = "application/vnd.github+json"
main_upstream_api_version = "2022-11-28"
main_upstream_api_headers = {
    "Accept": f"{main_upstream_api_responsetype}",
    "X-GitHub-Api-Version": f"{main_upstream_api_version}",
    "Authorization": f"Bearer {main_upstream_api_authtoken}",
}


def fetch_recipes(recipe_folder: str) -> List[str]:
    return [
        os.path.abspath(os.path.join(recipe_folder, file))
        for file in os.listdir(recipe_folder)
        if file.endswith(".yaml")
    ]


def fetch_scripts(script_folder: str) -> List[str]:
    script_files = []
    for root, dirs, files in os.walk(script_folder):
        script_files.extend(
            os.path.abspath(os.path.join(root, file))
            for file in files
            if file.endswith(".py")
        )
    return script_files


def fetch_files(file_folder: str) -> List[str]:
    return [
        os.path.abspath(os.path.join(file_folder, file))
        for file in os.listdir(file_folder)
    ]


def get_hashes(
    recipe_folder: str, script_folder: str, config_folder: str
) -> dict[str, str]:
    recipe_files = sorted(fetch_recipes(recipe_folder))
    script_files = sorted(fetch_scripts(script_folder))
    config_files = sorted(fetch_files(config_folder))

    script_hashes = {}
    for script_file in script_files:
        with open(script_file, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:6]
            script_hashes[script_file] = file_hash

    config_hashes = {}
    for config_file in config_files:
        with open(config_file, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:6]
            config_hashes[config_file] = file_hash

    # sort the hashes of the scripts and the configs
    script_hashes = dict(sorted(script_hashes.items()))
    config_hashes = dict(sorted(config_hashes.items()))

    # join the hashes of the scripts and the configs (the values of the dictionaries), make a new hash of the joined hashes
    merged_hashes = hashlib.sha256(
        "".join(list(script_hashes.values()) + list(config_hashes.values())).encode()
    ).hexdigest()[:6]

    hashes = {}
    for recipe_file in recipe_files:
        with open(recipe_file, "rb") as f:
            recipe_hash = hashlib.sha256(f.read()).hexdigest()[:6]
            if os.path.basename(recipe_file).split(".")[0] != "Scripts":
                hashes[recipe_file] = recipe_hash
                continue
            # add the merged hash to the recipe hash and make a new hash of the joined hashes
            file_hash = hashlib.sha256(
                (recipe_hash + merged_hashes).encode()
            ).hexdigest()[:6]
            hashes[recipe_file] = file_hash

    return hashes


if __name__ == "__main__":
    print("Start of container building process for ViroConstrictor")
    recipe_hashes = get_hashes(
        base_path_to_envs, base_path_to_scripts, base_path_to_configs
    )

    builtcontainers = []
    for recipe, VersionHash in recipe_hashes.items():
        # strip the name of the recipe to only get the name of the environment
        recipe_basename = os.path.basename(recipe).replace(".yaml", "")
        container_basename = f"{__prog__}_{recipe_basename}".lower()

        associated_container_def_file = os.path.join(
            base_path_to_container_defs, f"{recipe_basename}.dockerfile"
        )
        upstream_registry_url = (
            f"{main_upstream_registry}/{recipe_basename}:{VersionHash}"
        )
        upstream_existing_containers = (
            f"{main_upstream_api_enpoint}{__prog__}_{recipe_basename}/versions"
        )
        print(
            f"Checking if container '{container_basename}' with hash '{VersionHash}' exists in the upstream registry"
        )
        json_response = requests.get(
            upstream_existing_containers, headers=main_upstream_api_headers
        ).json()

        tags = []

        # if the container exists at all in the upstream registry, the json response will be a list.
        # If the container does not exist, the json response will be a dict with a message that the container does not exist.
        # You can therefore check if the json response is a list or a dict to see if the container exists or not.
        if isinstance(json_response, list):
            # json_response = json.loads(json_response)
            tags = [
                version["metadata"]["container"]["tags"] for version in json_response
            ]
            # flatten the list of tags
            tags = [tag for sublist in tags for tag in sublist]
            print(tags)

        if VersionHash in tags:
            print(
                f"Container '{container_basename}' with hash '{VersionHash}' already exists in the upstream registry"
            )
            continue

        print(
            f"Container '{container_basename}' with hash '{VersionHash}' does not exist in the upstream registry"
        )
        print(
            f"Starting Apptainer build process for container '{container_basename}:{VersionHash}'"
        )

        # create a temporary file to write the container definition to, copy the contents of {recipe_basename}.def to it and then append the labels section to it including the version hash
        # then use the temporary file as the container definition file for the apptainer build process
        # the apptainer build process will build the .sif container file also in a temporary directory
        # after the container is built, the built container file will be moved to the current working directory and the temporary directory will be deleted.
        # the container file will not be pushed to the upstream registry yet, this will be done in a separate script after all containers have been built and tested.
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False
        ) as tmp, tempfile.TemporaryDirectory() as tmpdir:
            with open(associated_container_def_file, "r") as f:
                tmp.write(f.read())
                tmp.write(
                    f"""

LABEL Author="RIVM-bioinformatics team"
LABEL Maintainer="RIVM-bioinformatics team"
LABEL Associated_pipeline="{__prog__}"
LABEL version="{VersionHash}"
LABEL org.opencontainers.image.authors="ids-bioinformatics@rivm.nl"
LABEL org.opencontainers.image.source=https://github.com/RIVM-bioinformatics/{__prog__}

    """
                )
            tmp.flush()  # flush the temporary file to make sure the contents are written to disk
            subprocess.run(
                [
                    "docker",
                    "build",
                    "-t",
                    f"{container_basename}:{VersionHash}",
                    "-f",
                    f"{tmp.name}",
                    ".",
                    "--network",
                    "host",
                    "--no-cache",
                ],
                check=True,
            )
            # move the container file to the current working directory
            subprocess.run(
                [
                    "docker",
                    "save",
                    "-o",
                    f"{base_path_to_container_defs}/{container_basename}:{VersionHash}.tar",
                    f"{container_basename}:{VersionHash}",
                ]
            )

        builtcontainers.append(f"{container_basename}:{VersionHash}")

    with open(f"{base_path_to_container_defs}/builtcontainers.json", "w") as f:
        json.dump(builtcontainers, f, indent=4)
