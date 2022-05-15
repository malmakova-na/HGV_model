import openmdao.api as om
from .mach_comp import MachComp
from .cl_comp import CLComp
from .dynamic_pressure_comp import DynamicPressureComp
from .lift_drag_force_comp import LiftDragForceComp
from .thrust_comp import ThrustComp


class AeroGroup(om.Group):
  
    def initialize(self):
        self.options.declare('num_nodes', types=int)
       
    def setup(self):
        nn = self.options['num_nodes']


        self.add_subsystem(name='mach_comp',
                           subsys=MachComp(num_nodes=nn),
                           promotes_inputs=['v', 'sos'],
                           promotes_outputs=['mach'])
        
        self.add_subsystem(name='cl_comp',
                           subsys=CLComp(num_nodes=nn),
                           promotes_inputs=['mach', 'alpha'],
                           promotes_outputs=['CL'])


        self.add_subsystem(name="q_comp",
                           subsys=DynamicPressureComp(num_nodes=nn),
                           promotes_inputs=['rho', 'v'],
                           promotes_outputs=['q'])

        self.add_subsystem(name='lift_drag_force_comp',
                           subsys=LiftDragForceComp(num_nodes=nn),
                           promotes_inputs=['CL', 'CD', 'q', 'S'],
                           promotes_outputs=['f_lift', 'f_drag'])
        
        self.add_subsystem(name='thrust_comp',
                           subsys=ThrustComp(num_nodes=nn),
                           promotes_inputs=['PLA', 'h'],
                           promotes_outputs=['T'])
