import os
import zipfile
import json

class SubmissionVerifier:
    def __init__(self, zip_path="submission.zip"):
        self.zip_path = zip_path

    def verify(self):
        print(f"🔬 VERIFYING SUBMISSION ARCHIVE '{self.zip_path}'...")
        
        if not os.path.exists(self.zip_path):
            print(f"❌ Error: Archive file '{self.zip_path}' not found!")
            return False

        # Check if it's a valid ZIP
        if not zipfile.is_zipfile(self.zip_path):
            print(f"❌ Error: '{self.zip_path}' is not a valid or uncorrupted ZIP archive!")
            return False

        errors = []
        warnings = []
        files_found = []
        
        try:
            with zipfile.ZipFile(self.zip_path, "r") as zipf:
                namelist = zipf.namelist()
                print(f"📋 Found {len(namelist)} files inside ZIP archive:")
                for name in namelist:
                    print(f"  - {name}")
                    files_found.append(name)

                # 1. Verify root-level directory constraints
                # Any nested file contains slashes like "folder/file"
                for name in namelist:
                    if "/" in name or "\\" in name:
                        errors.append(f"Nested directory path found: '{name}'. ALL files must be placed directly at the root of the ZIP archive!")

                # 2. Check required files
                config_name = "adapter_config.json"
                weights_names = ["adapter_model.safetensors", "adapter_model.bin"]
                
                if config_name not in files_found:
                    errors.append(f"Missing critical configuration file: '{config_name}'!")
                
                has_weights = False
                for wname in weights_names:
                    if wname in files_found:
                        has_weights = True
                        break
                if not has_weights:
                    errors.append("Missing trainable adapter weights file (must be adapter_model.safetensors or adapter_model.bin)!")

                # 3. Parse adapter_config.json and check rank constraint
                if config_name in files_found:
                    try:
                        config_bytes = zipf.read(config_name)
                        config_data = json.loads(config_bytes.decode("utf-8"))
                        
                        rank = config_data.get("r", 0)
                        print(f"🔎 Extracted LoRA configuration details:")
                        print(f"  - Target base model: {config_data.get('base_model_name_or_path', 'unknown')}")
                        print(f"  - LoRA Rank (r):     {rank}")
                        print(f"  - LoRA Alpha (alpha): {config_data.get('lora_alpha', 'unknown')}")
                        
                        if rank > 32:
                            errors.append(f"LoRA Rank Violation! Rank is r = {rank}, but competition rules STRICTLY limit it to at most 32!")
                        else:
                            print(f"  ✅ Rank constraint asserted: r = {rank} <= 32")
                            
                    except Exception as e:
                        errors.append(f"Failed to parse '{config_name}' as valid JSON: {e}")

        except Exception as e:
            errors.append(f"Error reading ZIP archive: {e}")

        # Compute package size
        size_mb = os.path.getsize(self.zip_path) / (1024 * 1024)
        
        print("\n" + "="*60)
        print("📊 KAGGLE COMPLIANCE VERIFICATION REPORT")
        print("="*60)
        print(f"Archive Path:       {self.zip_path}")
        print(f"Total Files:        {len(files_found)}")
        print(f"Archive Size:       {size_mb:.2f} MB")
        print(f"Compliance Status:  {'❌ FAILED' if errors else '✅ 100% COMPLIANT'}")
        print("-"*60)
        
        if errors:
            print(f"🚨 Errors Detected ({len(errors)}):")
            for err in errors:
                print(f"  - {err}")
            print("="*60)
            return False
        else:
            print("🎉 SUCCESS: Your 'submission.zip' package is 100% compliant with all Kaggle scoring constraints and is fully ready for submission!")
            print("="*60)
            return True

if __name__ == "__main__":
    verifier = SubmissionVerifier()
    verifier.verify()
