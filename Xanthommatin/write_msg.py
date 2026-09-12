import json
p = 'core/test_run_manifest.json'
m = json.load(open(p, 'r', encoding='utf-8'))
open('msg.bin', 'wb').write(bytes.fromhex(m['signature_hash']))
print('Wrote msg.bin', len(open('msg.bin','rb').read()), 'bytes')
