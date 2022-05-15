import matplotlib.pyplot as plt
import openmdao.api as om
import dymos as dm
import numpy as np
from dymos.examples.plotting import plot_results
from components.climb_ode import MinTimeClimbODE


if __name__ == "__main__":
    p = om.Problem(model=om.Group())

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'SLSQP'
    p.driver.declare_coloring()

    traj = dm.Trajectory()

    boost_phase = dm.Phase(ode_class=MinTimeClimbODE,
                        transcription=dm.Radau(num_segments=12))

    traj.add_phase('boost_phase', boost_phase) #Boost phase

    p.model.add_subsystem('traj', traj)

    boost_phase.set_time_options(fix_initial=True,duration_bounds=(0, 1000),
                        units= 's')


    boost_phase.add_state('r', fix_initial=True,lower=0, units='m',#lower=0, upper=6.0E7,
                    #ref=1.0E3, defect_ref=1.0E3,
                    rate_source='flight_dynamics.r_dot')

    boost_phase.add_state('h', fix_initial=True, #upper=60000,#units='m',#lower=6000,
                #ref=1.0E2, defect_ref=1.0E2,
                    rate_source='flight_dynamics.h_dot')

    boost_phase.add_state('v', fix_initial=True, lower=0.0, units='m/s',
                    #ref=1.0E2, defect_ref=1.0E2,
                    rate_source='flight_dynamics.v_dot')

    boost_phase.add_state('gam', fix_initial=True, # lower=-1.5, upper=1.5,
                    ref=1.0, defect_ref=1.0,
                    rate_source='flight_dynamics.gam_dot')

    boost_phase.add_control('alpha', units='deg',scaler=1.0,# upper=5, lower=-5, 
                    rate_continuity=True, rate_continuity_scaler=100.0,#100
                    rate2_continuity=False)


    boost_phase.add_parameter('S', val=100.3353, units='m**2', opt=False, targets=['S'])

    boost_phase.add_parameter('m', val=136077.711, units='kg', opt=False, targets=['m'])



    boost_phase.add_boundary_constraint('h', loc='final', equals=25000, scaler=1.0E-3, units='m')
    #phase.add_boundary_constraint('aero.mach', loc='final', equals=15.0)
    boost_phase.add_boundary_constraint('alpha', loc='final', lower=9, upper=12, units='deg')
    boost_phase.add_boundary_constraint('gam', loc='final', equals=np.deg2rad(-5), units='rad')
    boost_phase.add_boundary_constraint('v', loc='final', equals=1000)
    #boost_phase.add_path_constraint(name='aero.q', upper=90000)
    boost_phase.add_boundary_constraint('aero.q', loc='final', upper=90000)

    boost_phase.add_path_constraint(name='h',  upper=60000.0,lower = 2500, units='m')#, upper=60000.0
    #boost_phase.add_path_constraint(name='gam', lower=np.deg2rad(-5), upper=np.deg2rad(5))#вернуть
    boost_phase.add_path_constraint(name='v',  upper=5000) #lower=1000.0,
    #boost_phase.add_path_constraint(name='aero.q', upper=90000)

    #boost_phase.add_path_constraint(name='alpha', lower=np.deg2rad(-5), upper=np.deg2rad(5))

    # Maximize r at the end of the phase
    boost_phase.add_objective('r', loc='final', scaler=-1)


    p.model.linear_solver = om.DirectSolver()


    p.setup(check=True)

    p['traj.boost_phase.t_initial'] = 0
    p['traj.boost_phase.t_duration'] = 1000


    p.set_val('traj.boost_phase.states:r', boost_phase.interp('r', [0, 5000000.0]))#[500000, 5000000.0]
    p.set_val('traj.boost_phase.states:h', boost_phase.interp('h', [60000.0, 25000]))
    p.set_val('traj.boost_phase.states:v', boost_phase.interp('v', [5000, 1000]))
    p.set_val('traj.boost_phase.states:gam', boost_phase.interp('gam', [0.0, np.deg2rad(-5)]))
    p.set_val('traj.boost_phase.controls:alpha', boost_phase.interp('alpha', [1, 12]))#[0.0, 0.0]



    #boost_phase.add_timeseries_output('*')

    # Add parameters common to multiple phases to the trajectory

    # Solve for the optimal trajectory

    dm.run_problem(p, simulate=True)
    sol = om.CaseReader('dymos_solution.db').get_case('final')
    sim = om.CaseReader('dymos_simulation.db').get_case('final')
    plot_results([('traj.boost_phase.timeseries.states:r', 'traj.boost_phase.timeseries.states:h',
               'r (m)', 'altitude (m)')],
             title='Glide phase',
             p_sol=sol)

    plt.savefig("test_glade.png")
