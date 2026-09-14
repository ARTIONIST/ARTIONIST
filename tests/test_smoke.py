import asyncio, os
os.environ['MMI_PROVIDERS']='mock'
from mmi_psai_x1.app import investigate

def test_pipeline():
    x=asyncio.run(investigate('A system fails intermittently.'))
    assert x.run_id
    assert x.merged_hypotheses
    assert x.tests
    assert 0 <= float(x.judge['confidence']) <= 1
