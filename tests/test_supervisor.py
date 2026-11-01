import json
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch,MagicMock

from hvaclab import supervisor as module
from test_coverage import workspace_directory


class SupervisorTest(unittest.TestCase):
    def exercise(self,root):
        return patch.multiple(module,RUN=root)

    def test_existing_scheduler_is_adopted_without_duplicate_launch(self):
        with workspace_directory() as root,self.exercise(root), \
             patch.object(module,'study_processes',return_value=([SimpleNamespace(pid=42)],[])), \
             patch.object(module.subprocess,'Popen') as launch:
            self.assertTrue(module.Supervisor().tick())
            launch.assert_not_called()
            self.assertEqual(json.loads((root/'supervisor_status.json').read_text())['status'],'monitoring')

    def test_orphan_worker_finishes_before_restart(self):
        with workspace_directory() as root,self.exercise(root), \
             patch.object(module,'study_processes',return_value=([],[SimpleNamespace(pid=43)])), \
             patch.object(module.subprocess,'Popen') as launch:
            self.assertTrue(module.Supervisor().tick())
            launch.assert_not_called()
            self.assertEqual(json.loads((root/'supervisor_status.json').read_text())['status'],'waiting_for_orphan_workers')

    def test_failed_scheduler_restarts_after_backoff_and_stops_after_completion(self):
        with workspace_directory() as root,self.exercise(root), \
             patch.object(module,'study_processes',return_value=([],[])), \
             patch.object(module,'frozen_changes',return_value=[]), \
             patch.object(module.psutil,'virtual_memory',return_value=SimpleNamespace(available=4*2**30)), \
             patch('shutil.disk_usage',return_value=SimpleNamespace(free=4*2**30)), \
             patch.object(module.time,'monotonic',return_value=100) as clock, \
             patch.object(module.subprocess,'Popen') as launch:
            process=MagicMock(pid=44)
            process.poll.return_value=1
            launch.return_value=process
            runner=module.Supervisor()
            self.assertTrue(runner.tick())
            self.assertEqual(launch.call_count,1)
            clock.return_value=101
            self.assertTrue(runner.tick())
            self.assertEqual(launch.call_count,1)
            self.assertEqual(runner.failures,1)
            clock.return_value=120
            self.assertTrue(runner.tick())
            self.assertEqual(launch.call_count,2)
            process.poll.return_value=0
            (root/'status.json').write_text('{"status":"complete"}')
            self.assertFalse(runner.tick())
            self.assertEqual(launch.call_count,2)

    def test_changed_inputs_and_low_memory_do_not_launch(self):
        for changed,expected in ((['policy.py'],'blocked_inputs_changed'),([], 'waiting_for_memory')):
            with workspace_directory() as root,self.exercise(root), \
                 patch.object(module,'study_processes',return_value=([],[])), \
                 patch.object(module,'frozen_changes',return_value=changed), \
                 patch.object(module.psutil,'virtual_memory',return_value=SimpleNamespace(available=.5*2**30)), \
                 patch.object(module.subprocess,'Popen') as launch:
                module.Supervisor().tick()
                launch.assert_not_called()
                self.assertEqual(json.loads((root/'supervisor_status.json').read_text())['status'],expected)

    def test_stop_marker_leaves_existing_study_untouched(self):
        with workspace_directory() as root,self.exercise(root), \
             patch.object(module,'study_processes',return_value=([SimpleNamespace(pid=42)],[])), \
             patch.object(module.subprocess,'Popen') as launch:
            (root/'supervisor.stop').touch()
            self.assertFalse(module.Supervisor().tick())
            launch.assert_not_called()

    def test_repeated_failures_are_bounded(self):
        with workspace_directory() as root,self.exercise(root), \
             patch.object(module,'study_processes',return_value=([],[])), \
             patch.object(module,'frozen_changes',return_value=[]), \
             patch.object(module.subprocess,'Popen') as launch:
            runner=module.Supervisor()
            runner.failures=5
            self.assertTrue(runner.tick())
            launch.assert_not_called()
            self.assertEqual(json.loads((root/'supervisor_status.json').read_text())['status'],'blocked_repeated_failures')

    def test_real_child_crash_is_logged_and_next_child_completes(self):
        actual_popen=subprocess.Popen
        with workspace_directory() as root,self.exercise(root), \
             patch.object(module,'study_processes',return_value=([],[])), \
             patch.object(module,'frozen_changes',return_value=[]), \
             patch.object(module.psutil,'virtual_memory',return_value=SimpleNamespace(available=4*2**30)), \
             patch('shutil.disk_usage',return_value=SimpleNamespace(free=4*2**30)), \
             patch.object(module.time,'monotonic',return_value=100) as clock:
            calls=[]
            def worker(command,**kwargs):
                script="print('crash probe',flush=True); raise SystemExit(7)" if not calls else (
                    "from pathlib import Path; Path("+repr(str(root/'status.json'))+")"
                    ".write_text('{\"status\":\"complete\"}'); print('resume probe',flush=True)")
                proc=actual_popen([sys.executable,'-c',script],**kwargs)
                calls.append(proc)
                return proc
            runner=module.Supervisor()
            with patch.object(module.subprocess,'Popen',side_effect=worker):
                runner.tick()
                calls[0].wait(timeout=15)
                clock.return_value=101
                runner.tick()
                self.assertEqual(runner.failures,1)
                clock.return_value=120
                runner.tick()
                calls[1].wait(timeout=15)
                self.assertFalse(runner.tick())
            logs=''.join(p.read_text() for p in root.glob('scheduler_*.log'))
            self.assertIn('crash probe',logs)
            self.assertIn('resume probe',logs)
            events=[json.loads(line) for line in (root/'supervisor_events.jsonl').read_text().splitlines()]
            self.assertIn(7,[e.get('exit_code') for e in events])


if __name__=='__main__':unittest.main()
