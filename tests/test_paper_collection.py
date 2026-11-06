"""Private collector integrity tests with synthetic archive fixtures only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import uuid
from contextlib import contextmanager
import unittest
from unittest.mock import patch
import zipfile

SCRIPT=Path(__file__).resolve().parents[1]/'reports/scripts/collect_paper_results.py'
spec=importlib.util.spec_from_file_location('paper_collect',SCRIPT)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@contextmanager
def fixture_directory():
    root=Path(__file__).resolve().parents[1]/'experiments/runs/collection_test_fixtures'/uuid.uuid4().hex
    root.mkdir(parents=True)
    yield root
    # Retain tiny synthetic fixtures locally for inspection; never scan/delete home.


def design():
    buildings=[f'synthetic_{i}' for i in range(12)]
    protocol=dict(steps=8640,building_count=12,arms=list(module.ARMS),weather=list(module.WEATHER))
    tasks=[dict(building=b,weather=w,policy=a,steps=8640) for b in buildings for w in module.WEATHER for a in module.ARMS]
    return protocol,dict(buildings=buildings,tasks=tasks)


class CollectionTest(unittest.TestCase):
    def test_exact_full_grid_required(self):
        p,c=design();module.check_design(p,c)
        c['tasks'][-1]=c['tasks'][0]
        with self.assertRaises(ValueError):module.check_design(p,c)

    def test_short_episode_or_duplicate_building_rejected(self):
        p,c=design();c['tasks'][0]['steps']=864
        with self.assertRaises(ValueError):module.check_design(p,c)
        p,c=design();c['buildings'][1]=c['buildings'][0]
        with self.assertRaises(ValueError):module.check_design(p,c)

    def test_completion_and_protocol_must_match(self):
        s=dict(status='complete',completed=192,total=192,active=[])
        r=dict(status='complete_verified',protocol_sha256='abc')
        module.check_completion(s,r,'abc')
        with self.assertRaises(ValueError):module.check_completion(s,r,'wrong')
        s['completed']=191
        with self.assertRaises(ValueError):module.check_completion(s,r,'abc')

    def test_archive_hash_tampering_and_extra_file_rejected(self):
        with fixture_directory() as directory:
            p=Path(directory)/'bundle.zip';data=b'synthetic'
            m=dict(files_sha256={'record.json':hashlib.sha256(data).hexdigest()})
            for mode in ('valid','tampered','extra'):
                with zipfile.ZipFile(p,'w') as z:
                    z.writestr('record.json',data if mode!='tampered' else b'changed')
                    z.writestr('collection_manifest.json',json.dumps(m))
                    if mode=='extra':z.writestr('unexpected.txt',b'no')
                if mode=='valid':self.assertEqual(module.verify_bundle(p),m)
                else:
                    with self.assertRaises(ValueError):module.verify_bundle(p)

    def test_path_escape_rejected(self):
        for name in ('../secret','/home/secret','C:/secret','foo\\secret'):
            self.assertFalse(module.safe_member(name))
        self.assertTrue(module.safe_member('experiments/runs/job.result.json'))

    def test_repo_file_scope_checked(self):
        with fixture_directory() as directory:
            root=Path(directory);(root/'record.json').write_text('{}')
            self.assertEqual(module.contained(root,'record.json'),(root/'record.json').resolve())
            with self.assertRaises(ValueError):module.contained(root,'../outside')

    def test_collection_is_read_only_and_does_not_scan_secrets(self):
        with fixture_directory() as directory:
            root=Path(directory);(root/'result.json').write_text('{"fixture":"synthetic"}')
            (root/'.env').write_text('DO_NOT_COLLECT=synthetic-sentinel')
            original=module.sha(root/'result.json');output=root/'private.zip'
            with patch.object(module,'audit',return_value=({'result.json'},{'status':'synthetic_archive_stage_only'})),patch.object(module,'environment',return_value={'fixture':True}):
                module.collect(root,output)
            manifest=module.verify_bundle(output)
            self.assertNotIn('.env',manifest['files_sha256'])
            self.assertEqual(module.sha(root/'result.json'),original)
            with self.assertRaises(ValueError):module.collect(root,output)


if __name__=='__main__':unittest.main()
