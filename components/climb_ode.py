import openmdao.api as om
from .aero_group import AeroGroup
from dymos.models.atmosphere import USatm1976Comp
from .flight_path import FlightPath


class MinTimeClimbODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        
        
        self.add_subsystem(name='aero',
                           subsys=AeroGroup(num_nodes=nn),
                           promotes_inputs=['v','S', 'h', 'PLA'])
        

        self.add_subsystem(name='atmos',
                           subsys=USatm1976Comp(num_nodes=nn),
                           promotes_inputs=['h'])
        
        self.connect('atmos.sos', 'aero.sos')
        self.connect('atmos.rho', 'aero.rho')
        

        self.add_subsystem(name='flight_dynamics',
                           subsys=FlightPath(num_nodes=nn),
                           promotes_inputs=['m',  'v', 'gam', 'alpha']) 
        
        self.connect('aero.f_drag', 'flight_dynamics.D')
        self.connect('aero.f_lift', 'flight_dynamics.L')
        self.connect('aero.T', 'flight_dynamics.T')