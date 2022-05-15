import openmdao.api as om
import numpy as np


class FlightPath(om.ExplicitComponent):
    def initialize(self):
        """
        Задаем параметры компонента
        """
        self.options.declare('num_nodes', types=int)

    def setup(self):
        """
        Задаем входные и выходные переменные компонента
        """
        nn = self.options['num_nodes']
        
        self.add_input(name='m',
                       val=np.ones(nn),
                       desc='aircraft mass',
                       units='kg')

        self.add_input(name='v',
                       val=np.ones(nn),
                       desc='aircraft velocity magnitude',
                       units='m/s')

        self.add_input(name='T',
                       val=np.zeros(nn),
                       desc='thrust',
                       units='N')
        
        self.add_input(name='alpha',
                       val= np.zeros(nn),
                       desc='angle of attack',
                       units='rad')

        self.add_input(name='L',
                       val=np.zeros(nn),
                       desc='lift force',
                       units='N')

        self.add_input(name='D',
                       val=np.zeros(nn),
                       desc='drag force',
                       units='N')

        self.add_input(name='gam',
                       val=np.zeros(nn),
                       desc='flight path angle',
                       units='rad')

        self.add_output(name='v_dot',
                        val=np.zeros(nn),
                        desc='rate of change of velocity magnitude',
                        units='m/s**2')

        self.add_output(name='gam_dot',
                        val=np.zeros(nn),
                        desc='rate of change of flight path angle',
                        units='rad/s')

        self.add_output(name='h_dot',
                        val=np.zeros(nn),
                        desc='rate of change of altitude',
                        units='m/s')

        self.add_output(name='r_dot',
                        val=np.zeros(nn),
                        desc='rate of change of range',
                        units='m/s')
        
        "задаем связь между компонентами"
        ar = np.arange(nn)
        
        self.declare_partials('v_dot', 'T', rows=ar, cols=ar)
        self.declare_partials('v_dot', 'D', rows=ar, cols=ar)
        self.declare_partials('v_dot', 'm', rows=ar, cols=ar)
        self.declare_partials('v_dot', 'gam', rows=ar, cols=ar)
        self.declare_partials('v_dot', 'alpha', rows=ar, cols=ar)

        self.declare_partials('gam_dot', 'T', rows=ar, cols=ar)
        self.declare_partials('gam_dot', 'L', rows=ar, cols=ar)
        self.declare_partials('gam_dot', 'm', rows=ar, cols=ar)
        self.declare_partials('gam_dot', 'gam', rows=ar, cols=ar)
        self.declare_partials('gam_dot', 'alpha', rows=ar, cols=ar)
        self.declare_partials('gam_dot', 'v', rows=ar, cols=ar)

        self.declare_partials('h_dot', 'gam', rows=ar, cols=ar)
        self.declare_partials('h_dot', 'v', rows=ar, cols=ar)

        self.declare_partials('r_dot', 'gam', rows=ar, cols=ar)
        self.declare_partials('r_dot', 'v', rows=ar, cols=ar)
        
       
       
    def compute(self, inputs, outputs):
        """
        Вычисляем выходные переменные системы
        """
        g = 9.80665
        m = inputs['m']
        v = inputs['v']
        T = inputs['T']
        L = inputs['L']
        D = inputs['D']
        gam = inputs['gam']
        alpha = inputs['alpha']
        
        calpha = np.cos(alpha)
        salpha = np.sin(alpha)

        cgam = np.cos(gam)
        sgam = np.sin(gam)

        mv = m * v
        

        outputs['v_dot'] = (T * calpha - D) / m - g * sgam
        outputs['gam_dot'] = (T * salpha + L) / m - (g / v) * cgam
        outputs['h_dot'] = v * sgam
        outputs['r_dot'] = v * cgam
                

    def compute_partials(self, inputs, partials):
    
        g = 9.80665
        m = inputs['m']
        v = inputs['v']
        T = inputs['T']
        L = inputs['L']
        D = inputs['D']
        gam = inputs['gam']
       
        alpha = inputs['alpha']

        calpha = np.cos(alpha)
        salpha = np.sin(alpha)

        cgam = np.cos(gam)
        sgam = np.sin(gam)

        mv = m * v
        
        "Вычисляем производные выходных переменных"

        partials['v_dot', 'T'] = calpha / m
        partials['v_dot', 'D'] = -1.0 / m
        partials['v_dot', 'm'] = (D - T * calpha) / (m**2)
        partials['v_dot', 'gam'] = -g * cgam
        partials['v_dot', 'alpha'] = -T * salpha / m

        partials['gam_dot', 'T'] = salpha / mv
        partials['gam_dot', 'L'] = 1.0 / mv
        partials['gam_dot', 'm'] = -(L + T * salpha) / (m * mv)
        partials['gam_dot', 'gam'] = g * sgam / v
        partials['gam_dot', 'alpha'] = T * calpha / mv
        partials['gam_dot', 'v'] = g * cgam / v**2 - (L + T * salpha) / (v * mv)

        partials['h_dot', 'gam'] = v * cgam
        partials['h_dot', 'v'] = sgam

        partials['r_dot', 'gam'] = -v * sgam
        partials['r_dot', 'v'] = cgam
