import openmdao.api as om
import numpy as np


class MachComp(om.ExplicitComponent):
    """
    Расчет числа Маха
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Входные данные
        self.add_input(name='v', shape=(nn,), desc='velocity magnitude', units='m/s')
        self.add_input('sos', shape=(nn,), desc='speed of sound', units='m/s')
        
        # В
        self.add_output(name='mach', val=0.7*np.ones(nn), desc='Mach number', units=None)#0.7*
        # Jacobian
        ar = np.arange(nn)
        self.declare_partials(of='mach', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='mach', wrt='sos', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['mach'][:] = inputs['v'] / inputs['sos']

    def compute_partials(self, inputs, partials):
        partials['mach', 'v'] = 1.0 / (inputs['sos'])
        partials['mach', 'sos'] = -inputs['v'] / (inputs['sos'] ) ** 2