import openmdao.api as om
import numpy as np


class LiftDragForceComp(om.ExplicitComponent):
   
    # Задаем параметры компонента
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Входные данные
        self.add_input(name='CL', val=np.zeros(nn,), desc='lift coefficient', units=None)
        self.add_input(name='CD', val=10**(-4)*np.ones(nn,), desc='drag coefficient', units=None)
        self.add_input(name='q', val=np.zeros(nn,), desc='dynamic pressure', units='N/m**2')
        self.add_input(name='S', val=np.zeros(nn,), desc='aerodynamic reference area', units='m**2')

        # Выходные данные
        self.add_output(name='f_lift', shape=(nn,), desc='aerodynamic lift force', units='N')
        self.add_output(name='f_drag', shape=(nn,), desc='aerodynamic drag force', units='N')

        # Зависимость выходных переменных от входных данных
        ar = np.arange(nn)

        self.declare_partials(of='f_lift', wrt='q', dependent=True, rows=ar, cols=ar)
        self.declare_partials(of='f_lift', wrt='S', dependent=True, rows=ar, cols=ar)
        self.declare_partials(of='f_lift', wrt='CL', dependent=True, rows=ar, cols=ar)

        self.declare_partials(of='f_drag', wrt='q', dependent=True, rows=ar, cols=ar)
        self.declare_partials(of='f_drag', wrt='S', dependent=True, rows=ar, cols=ar)
        self.declare_partials(of='f_drag', wrt='CD', dependent=True, rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        q = inputs['q']
        S = inputs['S']
        CL = inputs['CL']
        CD = inputs['CD']

        qS = q * S
        
        # Вычисление выходных данных
        outputs['f_lift'] = qS* CL
        outputs['f_drag'] = q * CD

    def compute_partials(self, inputs, partials):
        q = inputs['q']
        S = inputs['S']
        CL = inputs['CL']
        CD = inputs['CD']

        qS = q * S
        
        # Вычисление производных выходной переменной CL
        partials['f_lift', 'q'] = S * CL
        partials['f_lift', 'S'] = q * CL
        partials['f_lift', 'CL'] = qS
        
        # Вычисление производных выходной переменной CD
        partials['f_drag', 'q'] = S * CD
        partials['f_drag', 'S'] = q * CD
        partials['f_drag', 'CD'] = qS