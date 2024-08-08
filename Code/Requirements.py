import subprocess
import sys

def install_package(package_name):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"Package '{package_name}' installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install package '{package_name}'. Error: {e}")

if __name__ == "__main__":
    package = "requests"
    install_package('metapub')
    install_package('openai')
    install_package('numpy')
    install_package('pandas')
    install_package('scikit-learn')
    install_package('openai')
    install_package('sentence-transformers')
    install_package('transformers')
    install_package('anthropic')
    install_package('google-generativeai')
    install_package('google-cloud-aiplatform')