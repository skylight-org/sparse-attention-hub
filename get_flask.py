from datasets import load_dataset
sbl = load_dataset('SWE-bench/SWE-bench_Lite', split='test')
flask_instances = [x['instance_id'] for x in sbl if 'flask' in x['repo'].lower()]
with open('flask_instances.txt', 'w') as f:
    for inst in flask_instances[:3]:
        f.write(inst + '\n')
print("Instances selected:", flask_instances[:3])
