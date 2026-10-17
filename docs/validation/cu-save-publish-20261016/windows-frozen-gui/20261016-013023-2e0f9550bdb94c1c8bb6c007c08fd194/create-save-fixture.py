import os
from fdm.models import ProjectState
from fdm.project_io import ProjectIO
from fdm.live_experiment import ExperimentSession, default_materials
p=ProjectState.empty()
s=ExperimentSession(sample_name="save-lifecycle-probe",save_images=False)
s.add_operator("probe")
s.set_materials(default_materials())
p.live_experiments.append(s)
ProjectIO.save(p, os.environ["FDM_SAVE_PROBE_PROJECT"])