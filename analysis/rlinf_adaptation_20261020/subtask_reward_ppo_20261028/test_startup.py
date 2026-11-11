"""Exercise the actual bootstrap environment assignment, without GPU imports."""
import ast
import json
from pathlib import Path
import unittest


class StartupTests(unittest.TestCase):
    def test_diagnostic_stage_cannot_replace_workflow_phase(self):
        source=Path(__file__).with_name('bootstrap.py').read_text()
        tree=ast.parse(source)
        expression=next(n for n in ast.walk(tree) if isinstance(n,ast.Call)
                        and isinstance(n.func,ast.Attribute) and n.func.attr=='update'
                        and isinstance(n.func.value,ast.Name) and n.func.value.id=='env')
        compiled=compile(ast.Expression(expression),'<actual bootstrap env.update>','eval')
        for task_phase in ('sample','train_eval','eval'):
            for stage in ('unpack_inline_patch','verify_reused_files','graphics_setup','start_driver'):
                namespace={'env':{},'config':{'phase':stage,'task_phase':task_phase,'rank':0,
                           'attempt':3,'job_name':'test','status_put':'test','result_put':'test','artifact_puts':{}},
                           'plan':{'seed':0,'code':'/code'},'base':Path('/base'),'root':Path('/job'),
                           'runtime':Path('/runtime'),'library_path':'/lib','vk':'/vk','egl':'/egl','json':json}
                eval(compiled,namespace)
                self.assertEqual(namespace['env']['B1K_PHASE'],task_phase)


if __name__=='__main__':unittest.main()
