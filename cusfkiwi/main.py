"""References

[1] - A.M.P. Tombs, The Extended Mars Lander, Exercise 1: Euler's equations and quaternions, September 18, 2013

"""

from multiprocessing.sharedctypes import Value
import numpy as np

class Simulation:
    """
    Class for running simulations and storing the resulting data.
    
    Args:
        bodies (list): Body object to add to the simulation.
        dt (float): Timestep.
        end_condition (callable): Function of the State. Must return True when the simulation needs to end.

    Attributes:
        states (list): List of State objects, containing the full state of the body at each timestep.
    """
    def __init__(self, body, dt, end_condition):
        self.body = body
        self.dt = dt
        self.end_condition = end_condition
        self.states = []

    def fdot(self, fn, t, debug = False):
        state = State.from_array(fn)

        # Linear motion
        force = np.array([0.0, 0.0, 0.0])

        for i in range(len(self.body.forces)):

            if self.body.forces[i].input == "none":
                extra_force = self.body.forces[i].value
            elif self.body.forces[i].input == "time":
                extra_force = self.body.forces[i].value(t)
            elif self.body.forces[i].input == "state":
                extra_force = self.body.forces[i].value(state)

            force += np.array(extra_force)

        if debug:
            print(f"total force = {force}")

        if self.body.mass.input == "none":
            mass = self.body.mass.value
        elif self.body.mass.input == "time":
            mass = self.body.mass.value(t)
        elif self.body.mass.input == "state":
            mass = self.body.mass.value(state)

        acc = force / mass

        # Rotational motion
        moment = np.array([0.0, 0.0, 0.0])

        for i in range(len(self.body.moments)):

            if self.body.moments[i].input == "none":
                extra_moment = self.body.moments[i].value
            elif self.body.moments[i].input == "time":
                extra_moment = self.body.moments[i].value(t)
            elif self.body.moments[i].input == "state":
                extra_moment = self.body.moments[i].value(state)

            moment += np.array(extra_moment)

        if debug:
            print(f"total moment = {moment}")

        if self.body.mass.input == "none":
            A, B, C = self.body.moments_of_inertia.value
        elif self.body.mass.input == "time":
            A, B, C = self.body.moments_of_inertia.value(t)
        elif self.body.mass.input == "state":
            A, B, C = self.body.moments_of_inertia.value(state)

        I = np.array([[A, 0.0, 0.0], [0.0, B, 0.0], [0.0, 0.0, C]])
        b = np.cross(I @ state.ang_vel, state.ang_vel) + moment

        w = np.array(state.ang_vel)
        wdot0 = (moment[0] + (B - C) * w[1] * w[2]) / A     # [1]
        wdot1 = (moment[1] + (C - A) * w[2] * w[0]) / B
        wdot2 = (moment[2] + (A - B) * w[0] * w[1]) / C
        wdot = np.array([wdot0, wdot1, wdot2])

        q = np.array(state.ang_pos)
        s = q[0]
        v = -q[1:]
        
        qdot0 = 0.5 * -np.dot(v, w)
        qdot123 = 0.5 * ( s*w + np.cross(v, w)) 
        qdot = np.array([qdot0, qdot123[0], qdot123[1], qdot123[2]])

        # Write all in a single vector
        vel = np.array(state.vel)
        return np.array([1, vel[0], vel[1], vel[2], acc[0], acc[1], acc[2], qdot[0], qdot[1], qdot[2], qdot[3], wdot[0], wdot[1], wdot[2]])

    def run(self, debug = False):
        # Pre-collect what we need to make code neater
        dt = self.dt
        fdot = self.fdot
        t = self.body.state.time

        while not self.end_condition(self.body.state):
            # RK4 implementation
            self.states.append(self.body.state)
            fn = self.body.state.to_array()

            if debug:
                print(self.body.state)

            k1 = fdot(fn = fn, t = t, debug = debug)

            if debug:
                print(f"k1 = {k1} \n")

            k2 = fdot(fn + k1*dt/2, t + dt/2)
            k3 = fdot(fn + k2*dt/2, t + dt/2)
            k4 = fdot(fn + k3*dt, t + dt)

            fnplusone = fn + (1/6)*(k1 + 2*k2 + 2*k3 + k4)*dt   # + O(dt^5) = [pos, vel]

            # Write current state
            self.body.state = State.from_array(fnplusone)

            t = t + dt   

    def x(self):
        """
        Returns:
            numpy.ndarray: x coordinates
        """
        array = np.zeros(len(self.states))

        for i in range(len(self.states)):
           array[i] =  self.states[i].pos[0]

        return array

    def y(self):
        """
        Returns:
            numpy.ndarray: y coordinates
        """
        array = np.zeros(len(self.states))

        for i in range(len(self.states)):
           array[i] =  self.states[i].pos[1]

        return array

    def z(self):
        """
        Returns:
            numpy.ndarray: z coordinates
        """
        array = np.zeros(len(self.states))

        for i in range(len(self.states)):
           array[i] =  self.states[i].pos[2]

        return array

    def t(self):
        """
        Returns:
            numpy.ndarray: Time values
        """
        array = np.zeros(len(self.states))

        for i in range(len(self.states)):
           array[i] =  self.states[i].time

        return array

    def V(self):
        """
        Returns:
            numpy.ndarray: Velocity magnitudes
        """
        array = np.zeros(len(self.states))

        for i in range(len(self.states)):
           array[i] =  np.linalg.norm(self.states[i].vel)

        return array

class Body:
    """
    Class for storing the inertial properties of a body as well as the forces and moments that are to be applied.

    Args:
        init_state (State): Initial conditions as a State object.
        mass (Mass): Mass object to represent the body's mass.
        moments_of_inertia (MomentsOfInertia): MomentsOfInertia object to represent the principal moments of inertia of the body.
        forces (list): List of Forces objects, to represent the forces on the body.
        moments (list): List of Moments objects, to represent the moments on the body.
    """
    def __init__(self, init_state, mass, moments_of_inertia, forces, moments):
        self.state = init_state
        self.mass = mass
        self.moments_of_inertia = moments_of_inertia
        self.forces = forces
        self.moments = moments

class State:
    """
    Class for storing the entire state of a body (6 degrees of freedom in total).

    Args:
        time (float): Time.
        pos (list): Position in the absolute reference frame, [x, y, z]. List or array of length 3.
        vel (list): Velocity in the absolute reference frame, [v_x, v_y, v_z]. List or array of length 3.
        ang_pos (list): Attitude in quaternion form, should be a length 4 list or array. Represents a conversion from the body reference frame to the absolute one. Can be useful to use scipy.spatial.transform.Rotation if you need help with this.
        ang_vel (list): Angular velocity in the body's reference frame, [w_A, w_B, w_C]. List or array of length 3.

    Attributes:
        time (float): Time.
        pos (list): Position in the absolute reference frame, [x, y, z].
        vel (list): Velocity in the absolute reference frame, [v_x, v_y, v_z].
        ang_pos (list): Attitude in quaternion form, should be a length 4 list or array. Represents a conversion from the body reference frame to the absolute one.
        ang_vel (list): Angular velocity in the body's reference frame, [w_A, w_B, w_C].
    """

    def __init__(self, time, pos, vel, ang_pos, ang_vel):
        self.time = time
        self.pos = pos
        self.vel = vel
        self.ang_pos = ang_pos
        self.ang_vel = ang_vel

    def __repr__(self):
        return f"time = {self.time} \npos = {self.pos} \nvel = {self.vel} \nang_pos = {self.ang_pos} \nang_vel = {self.ang_vel}"

    def to_array(self):
        time = self.time
        pos = self.pos
        vel = self.vel
        ang_pos = self.ang_pos
        ang_vel = self.ang_vel

        return np.array([time, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2], ang_pos[0], ang_pos[1], ang_pos[2], ang_pos[3], ang_vel[0], ang_vel[1], ang_vel[2]])

    @staticmethod
    def from_array(array):
        time = array[0]
        pos = array[1:4]
        vel = array[4:7]
        ang_pos = array[7:11]
        ang_vel = array[11:14]

        return State(time, pos, vel, ang_pos, ang_vel)

class Mass:
    """
    Class for representing the mass of a body.

    Args:
        value (float or callable): Mass. Either a constant or a callable.
        input (str, optional): "none" for a constant, "time" to receive the time as an input, "state" to receive a State object as input. Defaults to "none".
    """
    def __init__(self, value, input = "none"):
        self.value = value
        self.input = input

class MomentsOfInertia:
    """
    Class for representing the principal moments of inertia of a body. Note that these principal moments of inertia should be given in the body's reference frame. This makes them constant for a constant mass and geometry system.

    Args:
        value (list or callable): List of principal moments of inetia in the form [A, B, C]. Can be a constant, function of time, or function of the State.
        input (str, optional): "none" for a constant, "time" to receive the time as an input, "state" to receive a State object as input. Defaults to "none".
    """
    def __init__(self, value, input = "none"):
        self.value = value
        self.input = input

class Force:
    """
    Class for representing the forces on a body. These must be given in the absolute reference frame (NOT the body reference frame).

    Args:
        value (list or callable): Force, [F_x, F_y, F_z]. List or array of length 3. Should be in the absolute (NOT body) reference frame. Either a constant or a callable.
        input (str, optional): "none" for a constant, "time" to receive the time as an input, "state" to receive a State object as input. Defaults to "none".
    """
    def __init__(self, value, input = "none"):

        if input != "none" and input != "time" and input != "state":
            raise ValueError(f"'input' must be either 'none', 'time', or 'state'. You gave '{input}'.")

        self.value = value
        self.input = input

class Moment:
    """
    Class for representing the moments on a body. These must be given in the body's reference frame (NOT the absolute one).

    Args:
        value (list or callable): Moment, [M_A, M_B, M_C], representing components about the body's A, B and C axes (using a right hand rule). List or array of length 3. Should be in the body reference frame (NOT the absolute one). Either a constant or a callable.
        input (str, optional): "none" for a constant, "time" to receive the time as an input, "state" to receive a State object as input. Defaults to "none".
    """
    def __init__(self, value, input = "none"):
        
        if input != "none" and input != "time" and input != "state":
            raise ValueError(f"'input' must be either 'none', 'time', or 'state'. You gave '{input}'.")

        self.value = value
        self.input = input

