import openmdao.api as om
import numpy as np


class CLComp(om.ExplicitComponent):
    """ Computes the  lift coefficient
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Входные данные
        self.add_input('mach', shape=(nn,), desc='Mach number', units=None)
        self.add_input('alpha', shape=(nn,), desc='AOA', units='rad')

        # Выходные данные
        self.add_output(name='CL', val=np.zeros(nn), desc='lift coefficient', units=None)

        ar = np.arange(nn)
        self.declare_partials(of='CL', wrt='mach', rows=ar, cols=ar)
        self.declare_partials(of='CL', wrt='alpha', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        M = inputs['mach']
        alpha = inputs['alpha']
       
        outputs['CL'] = -8.19E-02 + 4.70E-02 * M + 1.86E-02 * alpha - 4.73E-04 * M * alpha - \
                        9.19E-03 * M**2-1.52E-04 * alpha**2+5.99E-07 * (alpha * M)**2 + \
                        7.74E-04 * M**3+4.08E-06 * alpha**3-2.93E-05 * M**4-3.91E-07 * alpha**4 + \
                        4.12E-07 * M**5 +1.30E-08 * alpha**5