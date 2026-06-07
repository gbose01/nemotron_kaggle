import os
import zipfile
import argparse

def main():
    parser = argparse.ArgumentParser(description="Package trained LoRA adapter into submission.zip")
    parser.add_argument("--source", type=str, default="outputs/dry_run_adapter", help="Directory containing the trained adapter")
    parser.add_argument("--output", type=str, default="submission.zip", help="Output zip path")
    args = parser.parse_args()

    print(f"📦 PACKAGING SUBMISSION FROM '{args.source}'...")
    
    if not os.path.exists(args.source):
        print(f"❌ Error: Source directory '{args.source}' does not exist!")
        return

    # Locate key files
    config_name = "adapter_config.json"
    weights_names = ["adapter_model.safetensors", "adapter_model.bin"]

    # Tokenizer / chat-template artifacts bundled if present so the scoring
    # engine reproduces the exact training-time chat format.
    optional_names = [
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "chat_template.jinja",
        "added_tokens.json",
    ]

    config_path = os.path.join(args.source, config_name)
    weights_path = None
    weights_name_used = None
    
    for wname in weights_names:
        wpath = os.path.join(args.source, wname)
        if os.path.exists(wpath):
            weights_path = wpath
            weights_name_used = wname
            break

    if not os.path.exists(config_path):
        print(f"❌ Error: Missing critical config file '{config_path}'!")
        return
        
    if not weights_path:
        print(f"❌ Error: Missing trainable weight files inside '{args.source}'!")
        return

    # Open zip file
    print(f"⏳ Archiving files to '{args.output}'...")
    try:
        bundled = [config_name, weights_name_used]
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Write files directly at the root (arcname forces root placement, omitting directory prefixes)
            zipf.write(config_path, arcname=config_name)
            zipf.write(weights_path, arcname=weights_name_used)

            for oname in optional_names:
                opath = os.path.join(args.source, oname)
                if os.path.exists(opath):
                    zipf.write(opath, arcname=oname)
                    bundled.append(oname)

        # Get size in MB
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        
        print("\n" + "="*60)
        print("🎉 SUBMISSION PACKAGE SUCCESSFULLY COMPILED!")
        print("="*60)
        print(f"Output File:    {args.output}")
        print(f"Package Size:   {size_mb:.2f} MB")
        print("Files Bundled (Root Level):")
        for fname in bundled:
            print(f"  - {fname}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error compiling submission package: {e}")

if __name__ == "__main__":
    main()
