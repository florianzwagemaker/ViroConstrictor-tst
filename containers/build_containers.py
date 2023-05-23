import os
from typing import List
import hashlib
import requests
import json
import tempfile
import subprocess
import shutil

from ViroConstrictor import __prog__

base_path_to_envs = "./ViroConstrictor/workflow/envs/"
base_path_to_scripts = "./ViroConstrictor/workflow/scripts/"
base_path_to_files = "./ViroConstrictor/workflow/files/"
base_path_to_container_defs = "./containers"
main_upstream_registry = "ghcr.io/rivm-bioinformatics"
main_upstream_api_enpoint = "https://api.github.com/orgs/RIVM-bioinformatics/packages/container/"
main_upstream_api_authtoken = os.environ.get("TOKEN")
main_upstream_api_responsetype = "application/vnd.github+json"
main_upstream_api_version = "2022-11-28"
main_upstream_api_headers = {
    "Accept": f"{main_upstream_api_responsetype}", 
    "X-GitHub-Api-Version": f"{main_upstream_api_version}",
    "Authorization": f"Bearer {main_upstream_api_authtoken}"
    }

def fetch_recipes(recipe_folder: str) -> List[str]:
    return [
        os.path.abspath(os.path.join(recipe_folder, file))
        for file in os.listdir(recipe_folder)
        if file.endswith(".yaml")
    ]


def get_hashes(recipe_folder: str) -> dict[str, str]:
    recipe_files = fetch_recipes(recipe_folder)
    hashes = {}
    for recipe_file in recipe_files:
        with open(recipe_file, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()[:6]
            hashes[recipe_file] = file_hash
    return hashes

if __name__ == "__main__":
    print("Start of container building process for ViroConstrictor")
    recipe_hashes = get_hashes(base_path_to_envs)

    mountscripts = [f"  {os.path.join(base_path_to_scripts, script)}    /scripts/{script}\n"
        for script in os.listdir(base_path_to_scripts)
    ]
    
    mountfiles = [f"    {os.path.join(base_path_to_files, file)}    /files/{file}\n" for file in os.listdir(base_path_to_files)]
    
    builtcontainers = []
    for recipe, VersionHash in recipe_hashes.items():
        # strip the name of the recipe to only get the name of the environment
        recipe_basename = os.path.basename(recipe).replace(".yaml", "")
        container_basename = f"{__prog__}_{recipe_basename}"
        associated_container_def_file = os.path.join(base_path_to_container_defs, f"{recipe_basename}.def")
        upstream_registry_url = f"{main_upstream_registry}/{recipe_basename}:{VersionHash}"
        upstream_existing_containers = f"{main_upstream_api_enpoint}{__prog__}_{recipe_basename}/versions"

        json_response = requests.get(upstream_existing_containers, headers=main_upstream_api_headers).json()

        tags = []

        # if the container exists at all in the upstream registry, the json response will be a list. 
        # If the container does not exist, the json response will be a dict with a message that the container does not exist.
        # You can therefore check if the json response is a list or a dict to see if the container exists or not.
        if isinstance(json_response, list):
            # json_response = json.loads(json_response)
            tags = [version["metadata"]["container"]["tags"] for version in json_response]
            #flatten the list of tags
            tags = [tag for sublist in tags for tag in sublist]


        if VersionHash in tags:
            print(f"Container '{container_basename}' with hash '{VersionHash}' already exists in the upstream registry")
            continue

        print(f"Container '{container_basename}' with hash '{VersionHash}' does not exist in the upstream registry")
        print(f"Starting Apptainer build process for container '{container_basename}:{VersionHash}'")

        # create a temporary file to write the container definition to, copy the contents of {recipe_basename}.def to it and then append the labels section to it including the version hash
        # then use the temporary file as the container definition file for the apptainer build process
        # the apptainer build process will build the .sif container file also in a temporary directory
        # after the container is built, the built container file will be moved to the current working directory and the temporary directory will be deleted.
        # the container file will not be pushed to the upstream registry yet, this will be done in a separate script after all containers have been built and tested.
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp, tempfile.TemporaryDirectory() as tmpdir:
            with open(associated_container_def_file, "r") as f:
                tmp.write(f.read())
                tmp.write(f"""
%labels
    Author RIVM-bioinformatics
    Associated_pipeline {__prog__}
    Version {VersionHash}

%files
{''.join(mountfiles)}
{''.join(mountscripts)}
    """)
            tmp.flush() # flush the temporary file to make sure the contents are written to disk
            subprocess.run(["sudo", "apptainer", "build", f"{tmpdir}/{container_basename}.sif", f"{tmp.name}"], check=True)
            #move the container file to the current working directory
            shutil.copyfile(f"{tmpdir}/{container_basename}.sif", f"{base_path_to_container_defs}/{container_basename}.sif")
        builtcontainers.append(f"{container_basename}.sif")
        print(tags)
        print(VersionHash)

        print(recipe_basename)
        print(container_basename)
    with open("builtcontainers.json", "w") as f:
        json.dump(builtcontainers, f, indent=4)