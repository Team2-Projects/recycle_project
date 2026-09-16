"""Compare stock Humble's known defect with the replacement process over DDS."""
import os
import pytest
from test_collision_models_ros import monitors  # noqa: F401

pytestmark = pytest.mark.skipif(os.environ.get('RECYCLE_ROS_PROBE') != '1',
                               reason='opt-in isolated ROS2 probe')


@pytest.mark.parametrize('model', ['approach', 'checked'])
def test_monitor_scan_loss_must_not_be_mistaken_for_clearance(monitors, model):
    with monitors(model) as p:
        assert p.sample((.08, 0.)) == (.08, 0.)
        assert p.sample((.08, 0.), [(0.32, -.4, .32, .4)]) == (0., 0.)
        assert p.sample((.08, 0.)) == (.08, 0.)
        before = p.guard_receipts
        velocity = p.sample((.08, 0.), [(0.32, -.4, .32, .4)], scan_on=False, duration=1.2)
        assert p.guard_receipts > before + 10  # Another subscriber still gets new obstacle scans.
        # Raw and responses remain live; only Monitor scan delivery is lost.
        assert p.last is not None
        if model == 'approach' and velocity != (0., 0.):
            pytest.xfail('Stock Humble 1.1.20 excludes timed-out scan and returns fresh nonzero speed')
        assert velocity == (0., 0.)
        if model == 'checked':
            assert not p.last.input_valid
            assert p.sample((.08, 0.)) == (.08, 0.)
            assert p.last.input_valid
