import openmdao.api as om
import numpy as np


class DynamicPressureComp(om.ExplicitComponent):
    # Задаем параметры компонента
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        # Входные данные
        self.add_input(name='rho',shape=(nn,), desc='atmospheric density', units='kg/m**3')
        self.add_input(name='v', shape=(nn,), desc='air-relative velocity', units='m/s')

        # Выходные данные
        self.add_output(name='q', shape=(nn,), desc='dynamic pressure', units='N/m**2')

        # Зависимость выходной переменной от входных данных
        ar = np.arange(nn)
        self.declare_partials(of='q', wrt='rho', rows=ar, cols=ar)
        self.declare_partials(of='q', wrt='v', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        rho = inputs['rho'] 
        
        # Вычисление выходных данных
        outputs['q'] = 0.5 * rho * (inputs['v'] ** 2)

    def compute_partials(self, inputs, partials):
        # Вычисление производных выходной переменной 
        partials['q', 'rho'] = 0.5 * inputs['v'] ** 2
        partials['q', 'v'] = inputs['rho'] * inputs['v'] 