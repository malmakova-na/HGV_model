import openmdao.api as om
import numpy as np


class ThrustComp(om.ExplicitComponent):
    """ Computes the  lift coefficient
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputsshape=(nn,)
        self.add_input('PLA',shape=(nn,), desc='pilot lever angle', units=None)#np.ones(nn,)
        self.add_input(name='h', val=np.zeros(nn,), desc='altitude', units='m')

        # Outputs
        self.add_output(name='T', val=np.zeros(nn), desc='thrust', units='N')

        # Jacobian
        ar = np.arange(nn)
        self.declare_partials(of='T', wrt='PLA', rows=ar, cols=ar)
        self.declare_partials(of='T', wrt='h', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        h = inputs['h']
        PLA = inputs['PLA']
        outputs['T'] = -79.7204 + 1.9927 * h + 316.2086 * PLA-79.7204 + 1.9927 * h + \
                       316.2086 * PLA -0.0126 * h**2 + 19.4536 * PLA**2
