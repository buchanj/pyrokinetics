from pyrokinetics import Pyro, PyroScan, PyroScan_LHD
import os
import numpy as np

# Point to input files
templates = os.path.join('..', 'templates')

# Template input file
gs2_file = os.path.join(templates, 'james_step.in')

# Load up pyro object
pyro = Pyro(gk_file=gs2_file, gk_type='GS2')

# Use existing parameter
param_1 = 'ky'
values_1 = np.arange(0.1, 0.3, 0.1)

# Add new parameter
param_2 = 'my_electron_gradient'
values_2 = np.arange(2.5, 4.0, 0.5)

# Dictionary of param and values
param_dict = {param_1: values_1,
              param_2: values_2}

# Create PyroScan object
pyro_scan = PyroScan_LHD(pyro,
                         directory = './test_GS2_LHD',
                         image_name = 'gs2_local',
                         template_file = gs2_gile,
                         param_dict,
                         value_fmt='.3f',
                         value_separator='_',
                         parameter_separator='_'
)

pyro_scan.add_parameter_key(param_2, 'local_species', ['electron', 'a_lt'])

pyro_scan.submit(npoints=248, max_containers=124, wait=False)
