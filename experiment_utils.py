import numpy as np
import openmdao.api as om
import os
import dymos as dm
import json
from components.climb_ode import MinTimeClimbODE
from dymos.examples.plotting import plot_results
from utils import plot_pair
from matplotlib import pyplot as plt


def run_phase(config, phase, save_path):
    p = om.Problem(model=om.Group())

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'SLSQP'
    p.driver.declare_coloring()

    traj = dm.Trajectory()

    if phase == "boost":
        num_segments = 6
    else:
        num_segments = 12

    boost_phase = dm.Phase(ode_class=MinTimeClimbODE,
                        transcription=dm.Radau(num_segments=num_segments))

    traj.add_phase('boost_phase', boost_phase) 

    p.model.add_subsystem('traj', traj)

    if phase == "boost":
        boost_phase.set_time_options(fix_initial=True, duration_bounds=(0, 250), duration_ref=200)
    else:
        boost_phase.set_time_options(fix_initial=True,duration_bounds=(0, 1000),
                        units= 's')

    if phase == "boost":
        boost_phase.add_state('r', fix_initial=True, lower=0, units='m', 
                        ref=1.0E3, defect_ref=1.0E3,
                        rate_source='flight_dynamics.r_dot')
    else:
        boost_phase.add_state('r', fix_initial=True, lower=0, units='m', 
                        rate_source='flight_dynamics.r_dot')

    print("add_state h upper: {}".format(config["path_constraints"]["height"]["upper"]))
    if phase == "boost":
        boost_phase.add_state('h', fix_initial=True, upper=config["path_constraints"]["height"]["upper"], units='m', 
                        ref=1.0E2, defect_ref=1.0E2,
                        rate_source='flight_dynamics.h_dot')
    else:
        boost_phase.add_state('h', fix_initial=True, 
                        rate_source='flight_dynamics.h_dot')

    if phase == "boost":
        boost_phase.add_state('v', fix_initial=True, lower=0.0, units='m/s',
                        ref=1.0E2, defect_ref=1.0E2,
                        rate_source='flight_dynamics.v_dot')
    else:
        boost_phase.add_state('v', fix_initial=True, lower=0.0, units='m/s',
                        rate_source='flight_dynamics.v_dot')

    boost_phase.add_state('gam', fix_initial=True,units='rad',
                    ref=1, defect_ref=1.0,
                    rate_source='flight_dynamics.gam_dot')

    if phase == "boost":
        boost_phase.add_control('alpha', units='deg',scaler=1.0,lower=0, upper=90,#fix_initial=False,
                        rate_continuity=True, rate_continuity_scaler=100.0,
                        rate2_continuity=False)
    else:
        boost_phase.add_control('alpha', units='deg',scaler=1.0,# upper=5, lower=-5, 
                    rate_continuity=True, rate_continuity_scaler=100.0,#100
                    rate2_continuity=False)

    boost_phase.add_parameter('S', val=100.3353, units='m**2', opt=False, targets=['S'])
    boost_phase.add_parameter('m', val=136077.711, units='kg', opt=False, targets=['m'])

    if phase == "boost":
        boost_phase.add_parameter('PLA', val=1.0,units=None, opt=False, targets=['PLA'])#upper=1.0,lower=0.0, 

    if phase == "glide":
        boost_phase.add_boundary_constraint('aero.q', loc='final', upper=90000)

    if "equals" in config["boundary_constraints"]["final"]["height"]:
        print("add_boundary_constraints h equals: {}".format(config["boundary_constraints"]["final"]["height"]["equals"]))
        boost_phase.add_boundary_constraint('h', loc='final', equals=config["boundary_constraints"]["final"]["height"]["equals"], units='m')
    elif "lower" in config["boundary_constraints"]["final"]["height"]:
        print("add_boundary_constraints h lower: {}".format(config["boundary_constraints"]["final"]["height"]["lower"]))
        print("add_boundary_constraints h upper: {}".format(config["boundary_constraints"]["final"]["height"]["upper"]))
        boost_phase.add_boundary_constraint('h', loc='final', lower=config["boundary_constraints"]["final"]["height"]["lower"], 
                                            upper=config["boundary_constraints"]["final"]["height"]["upper"], units='m')
    else:
        raise ValueError("miss boundary constraints for heigh")
    
    if "equals" in config["boundary_constraints"]["final"]["velocity"]:
        print("add_boundary_constraints v equals: {}".format(config["boundary_constraints"]["final"]["velocity"]["equals"]))
        boost_phase.add_boundary_constraint('v', loc='final', equals=config["boundary_constraints"]["final"]["velocity"]["equals"])
    elif "lower" in config["boundary_constraints"]["final"]["velocity"]:
        print("add_boundary_constraints v lower: {}".format(config["boundary_constraints"]["final"]["velocity"]["lower"]))
        print("add_boundary_constraints v upper: {}".format(config["boundary_constraints"]["final"]["velocity"]["upper"]))
        boost_phase.add_boundary_constraint('v', loc='final', lower=config["boundary_constraints"]["final"]["velocity"]["lower"],
                                            upper=config["boundary_constraints"]["final"]["velocity"]["upper"])
    else:
        raise ValueError("miss boundary constraints for heigh")
    
    if "equals" in config["boundary_constraints"]["final"]["alpha"]:
        print("add_boundary_constraints alpha equals: {}".format(config["boundary_constraints"]["final"]["alpha"]["equals"]))
        boost_phase.add_boundary_constraint('alpha', loc='final', equals=config["boundary_constraints"]["final"]["alpha"]["equals"], 
                                            units='deg')
    elif "lower" in config["boundary_constraints"]["final"]["alpha"]:
        print("add_boundary_constraints alpha lower: {}".format(config["boundary_constraints"]["final"]["alpha"]["lower"]))
        print("add_boundary_constraints alpha upper: {}".format(config["boundary_constraints"]["final"]["alpha"]["upper"]))
        boost_phase.add_boundary_constraint('alpha', loc='final', lower=config["boundary_constraints"]["final"]["alpha"]["lower"], 
                                            upper=config["boundary_constraints"]["final"]["alpha"]["upper"], units='deg')
    else:
        raise ValueError("miss boundary constraints for alpha")

    if "equals" in config["boundary_constraints"]["final"]["gamma"]:
        print("add_boundary_constraints gamma equals: {}".format(config["boundary_constraints"]["final"]["gamma"]["equals"]))
        boost_phase.add_boundary_constraint('gam', loc='final', equals=config["boundary_constraints"]["final"]["gamma"]["equals"])
    elif "lower" in config["boundary_constraints"]["final"]["gamma"]:
        print("add_boundary_constraints gamma lower: {}".format(config["boundary_constraints"]["final"]["gamma"]["lower"]))
        print("add_boundary_constraints gamma upper: {}".format(config["boundary_constraints"]["final"]["gamma"]["upper"]))
        boost_phase.add_boundary_constraint('gam', loc='final', lower=config["boundary_constraints"]["final"]["gamma"]["lower"],
                                            upper=config["boundary_constraints"]["final"]["gamma"]["upper"])
    else:
        raise ValueError("miss boundary constraints for gamma")

    if "height" in config["path_constraints"]:
        print("add_path_constraints h lower: {}".format(config["path_constraints"]["height"]["lower"]))
        print("add_path_constraints h upper: {}".format(config["path_constraints"]["height"]["upper"]))
        boost_phase.add_path_constraint(name='h', lower=config["path_constraints"]["height"]["lower"], 
                                        upper=config["path_constraints"]["height"]["upper"], units='m', ref=60000)
    else:
        print("Warning: no path constraints for height")
    
    if "velocity" in config["path_constraints"]:
        print("add_path_constraints v lower: {}".format(config["path_constraints"]["velocity"]["lower"]))
        print("add_path_constraints v upper: {}".format(config["path_constraints"]["velocity"]["upper"]))
        boost_phase.add_path_constraint(name='v',
                                        upper=config["path_constraints"]["velocity"]["upper"])
    else:
        print("Warning: no path constraints for velocity")

    # Minimize time at the end of the phase
    if phase == "boost":
        boost_phase.add_objective('time', loc='final', ref=1.0)
    else:
        boost_phase.add_objective('r', loc='final', scaler=-1)



    p.model.linear_solver = om.DirectSolver()


    p.setup(check=True)

    p['traj.boost_phase.t_initial'] = 0.0

    if phase == "boost":
        p['traj.boost_phase.t_duration'] = 250
    else:
        p['traj.boost_phase.t_duration'] = 1000


    p.set_val('traj.boost_phase.states:r', boost_phase.interp('r', [0.0, 500000.0]))
    p.set_val('traj.boost_phase.states:h', boost_phase.interp('h', [
        config["guesses"]["height"]["start"], config["guesses"]["height"]["end"]
    ]))
    p.set_val('traj.boost_phase.states:v', boost_phase.interp('v', [
        config["guesses"]["velocity"]["start"], config["guesses"]["velocity"]["end"]
    ]))
    p.set_val('traj.boost_phase.states:gam', boost_phase.interp('gam', [
        config["guesses"]["gamma"]["start"], config["guesses"]["gamma"]["end"]
    ]))
    p.set_val('traj.boost_phase.controls:alpha', boost_phase.interp('alpha', [
        config["guesses"]["alpha"]["start"], config["guesses"]["alpha"]["end"]
    ]))#


    # Solve for the optimal trajectory

    dm.run_problem(p, simulate=True)

    sol = om.CaseReader('dymos_solution.db').get_case('final')
    sim = om.CaseReader('dymos_simulation.db').get_case('final')

    r = p.get_val('traj.boost_phase.timeseries.states:r')
    r_km =r/1000

    h = p.get_val('traj.boost_phase.timeseries.states:h')
    h_km = h/1000

    t = p.get_val('traj.boost_phase.timeseries.time')
    v = p.get_val('traj.boost_phase.timeseries.states:v')

    if phase == "boost":
        title = "Фаза подъема"
    elif phase == "glide":
        title = "Фаза скольжения"

    sol = om.CaseReader('dymos_solution.db').get_case('final')
    sim = om.CaseReader('dymos_simulation.db').get_case('final')
    plot_results([('traj.boost_phase.timeseries.states:r', 'traj.boost_phase.timeseries.states:h',
               'r (m)', 'altitude (m)')],
             title='Glide phase',
             p_sol=sol)

    #plt.savefig("test_boost.png")
    
    os.makedirs(save_path, exist_ok=True)
    
    plot_pair(t, h_km, os.path.join(save_path, "{}_t_h".format(title)), "t(s)", "h(km)", title)
    plot_pair(t, v, os.path.join(save_path, "{}_t_v".format(title)), "t(s)", "v(m/s)", title)
    plot_pair(t, r_km, os.path.join(save_path, "{}_t_r".format(title)), "t(s)", "r(km)", title)
    plot_pair(r_km, h_km, os.path.join(save_path, "{}_r_h".format(title)), "r(km)", "h(km)", title)


def run_experiment(config, save_path):
    run_phase(config["boost"], "boost", save_path)
    run_phase(config["glide"], "glide", save_path)


if __name__ == "__main__":
    with open("configs/hgv.json", "r") as f:
        config = json.load(f)
    
    run_phase(config["glide"], "glide", "experiments/test")