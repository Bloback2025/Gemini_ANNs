## Verification workflow

1. Create a ledger copy for testing:
   - `Copy-Item core\ledger.log core\ledger.test.log -Force`

2. Generate and sign a manifest (use secure key path):
   - `python .\core\manifest_generator.py --state-file core\current_state.json --version <next> --previous-hash <prev> --private-key "C:\secure\signing_keys\ed25519_key.pem" --signer dev-local --out core\run_manifest.signed.json`

3. Verify the manifest with sentinel:
   - `python .\core\sentinel_copilot1.1.py --manifest core\run_manifest.signed.json --ledger core\ledger.test.log --pubkeys core\pubkeys --outdir core\verify_out_test --no-kms`

4. Archive artifacts on success:
   - `Copy-Item -Path core\verify_out_test\* -Destination core\archive -Recurse -Force`
   - Keep only public keys in `core/pubkeys`; never commit private keys.
