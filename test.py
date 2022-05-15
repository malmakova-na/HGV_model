import matplotlib.pyplot as plt
import openmdao.api as om
import dymos as dm
from dymos.examples.plotting import plot_results
from components.climb_ode import MinTimeClimbODE


if __name__ == "__main__":
#_______________________________________Boost______________________________________________________________
    p = om.Problem(model=om.Group())

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'SLSQP'
    p.driver.declare_coloring()

    traj = dm.Trajectory()

    boost_phase = dm.Phase(ode_class=MinTimeClimbODE,
                        transcription=dm.Radau(num_segments=6))

    traj.add_phase('boost_phase', boost_phase) 

    p.model.add_subsystem('traj', traj)

    boost_phase.set_time_options(fix_initial=True, duration_bounds=(0, 250), duration_ref=200)

    boost_phase.add_state('r', fix_initial=True, lower=0, units='m', 
                    ref=1.0E3, defect_ref=1.0E3,
                    rate_source='flight_dynamics.r_dot')

    boost_phase.add_state('h', fix_initial=True, upper=60000.0, units='m', 
                    ref=1.0E2, defect_ref=1.0E2,
                    rate_source='flight_dynamics.h_dot')

    boost_phase.add_state('v', fix_initial=True, lower=0.0, units='m/s',
                    ref=1.0E2, defect_ref=1.0E2,
                    rate_source='flight_dynamics.v_dot')

    boost_phase.add_state('gam', fix_initial=True,units='rad',
                    ref=1, defect_ref=1.0,
                    rate_source='flight_dynamics.gam_dot')


    boost_phase.add_control('alpha', units='deg',scaler=1.0,lower=0, upper=90,#fix_initial=False,
                      rate_continuity=True, rate_continuity_scaler=100.0,
                      rate2_continuity=False)

    boost_phase.add_parameter('S', val=100.3353, units='m**2', opt=False, targets=['S'])
    boost_phase.add_parameter('m', val=136077.711, units='kg', opt=False, targets=['m'])
    boost_phase.add_parameter('PLA', val=1.0,units=None, opt=False, targets=['PLA'])#upper=1.0,lower=0.0, 



    boost_phase.add_boundary_constraint('h', loc='final', equals=60000, units='m')#,scaler=1.0E-3
    boost_phase.add_boundary_constraint('v', loc='final', equals=5000)
    boost_phase.add_boundary_constraint('alpha', loc='final', lower=1, upper=12, units='deg')
    boost_phase.add_boundary_constraint('gam', loc='final',equals = 0)
    #boost_phase.add_boundary_constraint('aero.q', loc='final',upper=90000)

    boost_phase.add_path_constraint(name='h', lower=6000.0, upper=60000, units='m', ref=60000)
    boost_phase.add_path_constraint(name='v', upper=5000)# lower=500.0,

    # Minimize time at the end of the phase
    boost_phase.add_objective('time', loc='final', ref=1.0)



    p.model.linear_solver = om.DirectSolver()


    p.setup(check=True)

    p['traj.boost_phase.t_initial'] = 0.0
    p['traj.boost_phase.t_duration'] = 250


    p.set_val('traj.boost_phase.states:r', boost_phase.interp('r', [0.0, 500000.0]))
    p.set_val('traj.boost_phase.states:h', boost_phase.interp('h', [6000,60000]))
    p.set_val('traj.boost_phase.states:v', boost_phase.interp('v', [500,5000]))
    p.set_val('traj.boost_phase.states:gam', boost_phase.interp('gam', [0.0, 0.0]))
    p.set_val('traj.boost_phase.controls:alpha', boost_phase.interp('alpha', [1,12]))#


    # Solve for the optimal trajectory

    dm.run_problem(p, simulate=True)
    sol = om.CaseReader('dymos_solution.db').get_case('final')
    sim = om.CaseReader('dymos_simulation.db').get_case('final')
    plot_results([('traj.boost_phase.timeseries.states:r','traj.boost_phase.timeseries.states:h', 
               'r(m)', 'h(m)')],
             title='Hypersonic Glide Phase',
             p_sol=sol)

    plt.savefig("test_boost.png")
