"""
Environment test script to verify the installation of required libraries
and compatibility with the system configuration.
"""
import os
import sys
import importlib
import subprocess
import platform
from typing import List, Dict, Tuple

import torch
import torch
import torch_geometric
import torch_scatter
import torch_sparse

print(f"PyTorch: {torch.__version__}")
print(f"PyG: {torch_geometric.__version__}")
print(f"torch-scatter: {torch_scatter.__version__}")
print(f"torch-sparse: {torch_sparse.__version__}")


print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)

def print_result(name: str, status: bool, version: str = None, message: str = None) -> None:
    """Print the test result in a formatted way."""
    status_str = "✅ PASS" if status else "❌ FAIL"
    version_str = f"v{version}" if version else ""
    message_str = f" - {message}" if message else ""
    
    print(f"{status_str} | {name:20} | {version_str:10} {message_str}")

def get_package_version(package_name: str) -> str:
    """Get the version of an installed package."""
    try:
        return importlib.__import__(package_name).__version__
    except (AttributeError, ModuleNotFoundError):
        try:
            # Some packages store version differently
            return importlib.__import__(package_name.replace('-', '_')).version
        except (AttributeError, ModuleNotFoundError):
            return "unknown"

def check_package(package_name: str) -> Tuple[bool, str, str]:
    """Check if a package is installed and get its version."""
    try:
        importlib.import_module(package_name.replace('-', '_'))
        version = get_package_version(package_name)
        return True, version, ""
    except ModuleNotFoundError:
        return False, "", f"Package not found. Try: pip install {package_name}"
    except Exception as e:
        return False, "", str(e)

def check_gpu() -> Tuple[bool, str]:
    """Check for GPU availability with PyTorch."""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            return True, f"Found {device_count} GPU(s): {device_name}"
        else:
            return False, "No GPU detected by PyTorch"
    except Exception as e:
        return False, str(e)

def check_faiss_gpu() -> Tuple[bool, str]:
    """Check for FAISS GPU support."""
    try:
        import faiss
        has_gpu = hasattr(faiss, 'index_cpu_to_gpu')
        if has_gpu:
            return True, "FAISS with GPU support is available"
        else:
            return False, "FAISS is installed but without GPU support"
    except ModuleNotFoundError:
        return False, "FAISS is not installed"
    except Exception as e:
        return False, str(e)

def check_pytorch_geometric() -> Tuple[bool, str]:
    """Check PyTorch Geometric installation and compatibility."""
    try:
        import torch_geometric
        import torch
        # Check if PyG and PyTorch versions are compatible
        torch_version = torch.__version__.split('+')[0]
        pyg_version = torch_geometric.__version__
        message = f"PyTorch {torch_version}, PyG {pyg_version}"
        return True, message
    except Exception as e:
        return False, str(e)

def check_import_and_basic_functionality(module_name: str, check_fn) -> Tuple[bool, str]:
    """Import a module and check basic functionality."""
    try:
        module = importlib.import_module(module_name)
        return check_fn(module)
    except Exception as e:
        return False, str(e)

def check_huggingface_transformers(module) -> Tuple[bool, str]:
    """Verify Hugging Face transformers functionality."""
    try:
        # Check if we can access pretrained models configuration
        from transformers import AutoConfig
        AutoConfig.from_pretrained("bert-base-uncased", trust_remote_code=False)
        return True, "Successfully loaded model configuration"
    except Exception as e:
        return False, str(e)

def check_sentence_transformers(module) -> Tuple[bool, str]:
    """Verify sentence-transformers functionality."""
    try:
        # Try to load a small model just for testing
        module.SentenceTransformer('paraphrase-MiniLM-L3-v2')
        return True, "Successfully initialized model"
    except Exception as e:
        return False, str(e)

def check_openai(module) -> Tuple[bool, str]:
    """Verify OpenAI package functionality."""
    has_key = "OPENAI_API_KEY" in os.environ
    key_msg = "API key found" if has_key else "No API key found in environment"
    return True, key_msg

def check_requirements_file(filename: str = "requirements.txt") -> List[Tuple[str, bool, str]]:
    """Check all packages listed in the requirements file."""
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        packages = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or ';' in line:
                continue
            # Extract package name, ignoring version specifiers
            package = line.split('>=')[0].split('==')[0].split('>')[0].split('<')[0].strip()
            packages.append(package)
        
        results = []
        for package in packages:
            success, version, message = check_package(package)
            results.append((package, success, version))
        
        return results
    except Exception as e:
        print(f"Error checking requirements file: {e}")
        return []

def main():
    """Run all environment tests."""
    print_header("System Information")
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    
    # Check GPU availability
    print_header("GPU Configuration")
    gpu_available, gpu_message = check_gpu()
    print_result("CUDA GPU", gpu_available, message=gpu_message)
    
    faiss_gpu, faiss_message = check_faiss_gpu()
    print_result("FAISS GPU", faiss_gpu, message=faiss_message)
    
    # Check key ML libraries
    print_header("Key Libraries")
    libs_to_check = [
        ("torch", lambda m: (True, f"PyTorch {m.__version__}")),
        ("torch_geometric", lambda m: check_pytorch_geometric()),
        ("transformers", check_huggingface_transformers),
        ("sentence_transformers", check_sentence_transformers),
        ("openai", check_openai),
    ]
    
    for lib_name, check_fn in libs_to_check:
        success, version, _ = check_package(lib_name)
        if success:
            func_success, func_message = check_import_and_basic_functionality(lib_name, check_fn)
            print_result(lib_name, func_success, version, func_message)
        else:
            print_result(lib_name, False, message="Not installed")
    
    # Check all requirements
    print_header("Requirements Check")
    if os.path.exists("requirements.txt"):
        print("Checking packages from requirements.txt:")
        pkg_results = check_requirements_file()
        
        # Print summary
        failed = [pkg for pkg, success, _ in pkg_results if not success]
        if failed:
            print(f"\n⚠️  {len(failed)} packages not found: {', '.join(failed)}")
        else:
            print("\n✅ All required packages are installed!")
    else:
        print("requirements.txt not found")

if __name__ == "__main__":
    main()