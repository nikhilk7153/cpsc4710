from sae_lens import SAE

# Try to access available releases
try:
    # This might show us what's available
    from sae_lens.pretrained import list_sae_releases
    releases = list_sae_releases()
    print("Available SAE releases:")
    for r in releases:
        print(f"  - {r}")
except:
    print("Could not list releases directly")
    print("\nTrying common Llama-3 release names...")
    
    # Try common patterns
    possible_names = [
        "llama-3-8b",
        "llama3-8b",
        "meta-llama-3-8b",
        "llama-3-8b-instruct",
        "gpt2-small-res-jb",  # Known working release
    ]
    
    for name in possible_names:
        try:
            print(f"\nTrying: {name}")
            # Just check if it exists
            from sae_lens.sae import get_pretrained_saes_directory
            result = get_pretrained_saes_directory()
            if name in result:
                print(f"  ✓ Found: {name}")
                if name in result:
                    print(f"    SAE IDs: {list(result[name].keys())[:5]}")
        except Exception as e:
            print(f"  ✗ Error: {e}")
