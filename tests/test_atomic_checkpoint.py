from pathlib import Path
import unittest
from unittest.mock import patch
from hvaclab.coverage import replace_checkpoint
from test_coverage import workspace_directory


def blocked(code):
    error=PermissionError('sharing/lock test');error.winerror=code;return error


class AtomicCheckpointTest(unittest.TestCase):
    def test_transient_windows_lock_retries_and_preserves_old_checkpoint_until_success(self):
        with workspace_directory() as root:
            target=root/'status.json';temporary=root/'status.json.tmp'
            target.write_text('old');temporary.write_text('new');original=Path.replace;count=[0]
            def replace(path,destination):
                count[0]+=1
                if count[0]<3:
                    self.assertEqual(target.read_text(),'old');raise blocked(32)
                return original(path,destination)
            with patch.object(Path,'replace',autospec=True,side_effect=replace),patch('hvaclab.coverage.time.sleep') as sleep:
                replace_checkpoint(temporary,target)
                self.assertEqual(sleep.call_count,2)
            self.assertEqual(target.read_text(),'new');self.assertFalse(temporary.exists())

    def test_permanent_windows_denial_is_bounded_and_does_not_destroy_checkpoint(self):
        with workspace_directory() as root:
            target=root/'status.json';temporary=root/'status.json.tmp';target.write_text('old');temporary.write_text('new')
            with patch.object(Path,'replace',side_effect=blocked(5)) as replace,patch('hvaclab.coverage.time.sleep') as sleep:
                with self.assertRaises(PermissionError):replace_checkpoint(temporary,target)
                self.assertEqual(replace.call_count,7);self.assertEqual(sleep.call_count,6)
            self.assertEqual(target.read_text(),'old');self.assertTrue(temporary.exists())

    def test_unrelated_errors_are_not_silently_retried(self):
        with patch.object(Path,'replace',side_effect=FileNotFoundError('missing')),patch('hvaclab.coverage.time.sleep') as sleep:
            with self.assertRaises(FileNotFoundError):replace_checkpoint('missing','target')
            sleep.assert_not_called()


if __name__=='__main__':unittest.main()
