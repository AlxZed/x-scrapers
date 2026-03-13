import os
import subprocess
import sys


def install_writing_system():
    token = os.environ.get("GITUB_TOKEN_WRITING")
    if not token:
        raise ValueError("GITUB_TOKEN_WRITING not found in environment variables")

    # Uninstall
    print("Uninstalling writing-system")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "uninstall", "writing-system", "-y",
            "--break-system-packages",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Install
    repo_url = f"git+https://{token}@github.com/AlxZed/ai_writing_system.git"
    print("Installing writing-system")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--upgrade",
            "--break-system-packages", repo_url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


if __name__ == "__main__":
    install_writing_system()
