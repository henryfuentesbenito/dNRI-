import numpy as np
import matplotlib.pyplot as plt
import time


class SpringSim(object):
    def __init__(
        self,
        n_balls=5,
        box_size=5.0,
        loc_std=0.5,
        vel_norm=0.5,
        interaction_strength=0.1,
        noise_var=0.0,
    ):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._spring_types = np.array([0.0, 0.5, 1.0])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

    def _energy(self, loc, vel, edges):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide="ignore"):
            K = 0.5 * (vel**2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r**2).sum())
                        U += (
                            0.5
                            * self.interaction_strength
                            * edges[i, j]
                            * (dist**2)
                            / 2
                        )
            return U + K

    def _clamp(self, loc, vel):
        """
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        """
        assert np.all(loc < self.box_size * 3)
        assert np.all(loc > -self.box_size * 3)

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert np.all(loc <= self.box_size)

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert np.all(loc >= -self.box_size)
        vel[under] = np.abs(vel[under])

        return loc, vel

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A**2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B**2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def sample_trajectory(
        self, T=10000, sample_freq=10, spring_prob=[1.0 / 2, 0, 1.0 / 2]
    ):
        n = self.n_balls
        assert T % sample_freq == 0
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        # Sample edges
        edges = np.random.choice(
            self._spring_types, size=(self.n_balls, self.n_balls), p=spring_prob
        )
        edges = np.tril(edges) + np.tril(edges, -1).T
        np.fill_diagonal(edges, 0)
        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next**2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide="ignore"):
            forces_size = -self.interaction_strength * edges
            np.fill_diagonal(
                forces_size, 0
            )  # self forces are zero (fixes division by zero)
            F = (
                forces_size.reshape(1, n, n)
                * np.concatenate(
                    (
                        np.subtract.outer(loc_next[0, :], loc_next[0, :]).reshape(
                            1, n, n
                        ),
                        np.subtract.outer(loc_next[1, :], loc_next[1, :]).reshape(
                            1, n, n
                        ),
                    )
                )
            ).sum(axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                forces_size = -self.interaction_strength * edges
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (
                    forces_size.reshape(1, n, n)
                    * np.concatenate(
                        (
                            np.subtract.outer(loc_next[0, :], loc_next[0, :]).reshape(
                                1, n, n
                            ),
                            np.subtract.outer(loc_next[1, :], loc_next[1, :]).reshape(
                                1, n, n
                            ),
                        )
                    )
                ).sum(axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            return loc, vel, edges



"""La clase SpringSim simula un sistema de partículas conectadas por resortes y proporciona métodos para calcular las trayectorias 
de las partículas en función de sus interacciones a lo largo del tiempo.

Atributos
n_balls: Número de partículas en la simulación.
box_size: Tamaño de la caja que contiene las partículas.
loc_std: Desviación estándar de las posiciones iniciales de las partículas.
vel_norm: Norma de la velocidad inicial de las partículas.
interaction_strength: Fuerza de interacción entre las partículas.
noise_var: Varianza del ruido añadido a las observaciones.
_spring_types: Tipos de resortes con sus respectivas constantes.
_delta_T: Paso de tiempo para la simulación.
_max_F: Fuerza máxima permitida basada en el paso de tiempo.
Métodos
__init__: Inicializa la simulación con los parámetros dados y configura los tipos de resortes, el paso de tiempo y la fuerza máxima.

_energy: Calcula la energía total del sistema (cinética y potencial) basada en las posiciones y velocidades de las partículas y las conexiones entre ellas.

_clamp: Asegura que las partículas permanezcan dentro de los límites de la caja. Ajusta las posiciones y velocidades para simular colisiones elásticas con las paredes de la caja.

_l2: Calcula la distancia L2 (euclidiana) entre cada par de filas en dos matrices. Esto se utiliza para determinar las distancias entre partículas.

sample_trajectory: Genera una trayectoria de muestra para las partículas en la simulación. Este método:

Inicializa las posiciones y velocidades de las partículas.
Simula la dinámica del sistema durante un tiempo T, actualizando las posiciones y velocidades en cada paso de tiempo.
Añade ruido a las observaciones.
Devuelve las posiciones, velocidades y conexiones entre las partículas a lo largo del tiempo.
Utilidad
La clase SpringSim es útil para simular sistemas físicos donde las partículas interactúan a través de fuerzas de resorte,
como en modelos de moléculas o sistemas de masas y resortes. La simulación puede ser utilizada para estudiar el comportamiento 
dinámico de estos sistemas y generar datos para entrenar modelos de aprendizaje automático que infieren dinámicas físicas."""




class ChargedParticlesSim(object):
    def __init__(
        self,
        n_balls=5,
        box_size=5.0,
        loc_std=1.0,
        vel_norm=0.5,
        interaction_strength=1.0,
        noise_var=0.0,
    ):
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var

        self._charge_types = np.array([-1.0, 0.0, 1.0])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A**2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B**2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def _energy(self, loc, vel, edges):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide="ignore"):
            K = 0.5 * (vel**2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r**2).sum())
                        U += 0.5 * self.interaction_strength * edges[i, j] / dist
            return U + K

    def _clamp(self, loc, vel):
        """
        :param loc: 2xN location at one time stamp
        :param vel: 2xN velocity at one time stamp
        :return: location and velocity after hiting walls and returning after
            elastically colliding with walls
        """
        assert np.all(loc < self.box_size * 3)
        assert np.all(loc > -self.box_size * 3)

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert np.all(loc <= self.box_size)

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert np.all(loc >= -self.box_size)
        vel[under] = np.abs(vel[under])

        return loc, vel

    def sample_trajectory(
        self, T=10000, sample_freq=10, charge_prob=[1.0 / 2, 0, 1.0 / 2]
    ):
        n = self.n_balls
        assert T % sample_freq == 0
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        # Sample edges
        charges = np.random.choice(
            self._charge_types, size=(self.n_balls, 1), p=charge_prob
        )
        edges = charges.dot(charges.transpose())
        # Initialize location and velocity
        loc = np.zeros((T_save, 2, n))
        vel = np.zeros((T_save, 2, n))
        loc_next = np.random.randn(2, n) * self.loc_std
        vel_next = np.random.randn(2, n)
        v_norm = np.sqrt((vel_next**2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide="ignore"):
            # half step leapfrog
            l2_dist_power3 = np.power(
                self._l2(loc_next.transpose(), loc_next.transpose()), 3.0 / 2.0
            )

            # size of forces up to a 1/|r| factor
            # since I later multiply by an unnormalized r vector
            forces_size = self.interaction_strength * edges / l2_dist_power3
            np.fill_diagonal(
                forces_size, 0
            )  # self forces are zero (fixes division by zero)
            assert np.abs(forces_size[diag_mask]).min() > 1e-10
            F = (
                forces_size.reshape(1, n, n)
                * np.concatenate(
                    (
                        np.subtract.outer(loc_next[0, :], loc_next[0, :]).reshape(
                            1, n, n
                        ),
                        np.subtract.outer(loc_next[1, :], loc_next[1, :]).reshape(
                            1, n, n
                        ),
                    )
                )
            ).sum(axis=-1)
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                l2_dist_power3 = np.power(
                    self._l2(loc_next.transpose(), loc_next.transpose()), 3.0 / 2.0
                )
                forces_size = self.interaction_strength * edges / l2_dist_power3
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)

                F = (
                    forces_size.reshape(1, n, n)
                    * np.concatenate(
                        (
                            np.subtract.outer(loc_next[0, :], loc_next[0, :]).reshape(
                                1, n, n
                            ),
                            np.subtract.outer(loc_next[1, :], loc_next[1, :]).reshape(
                                1, n, n
                            ),
                        )
                    )
                ).sum(axis=-1)
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, 2, self.n_balls) * self.noise_var
            return loc, vel, edges




"""La clase ChargedParticlesSim simula un sistema de partículas cargadas que interactúan mediante fuerzas de Coulomb 
(atracción y repulsión). Proporciona métodos para calcular las trayectorias de las partículas en función de sus interacciones a lo largo del tiempo.

Atributos
n_balls: Número de partículas en la simulación.
box_size: Tamaño de la caja que contiene las partículas.
loc_std: Desviación estándar de las posiciones iniciales de las partículas.
vel_norm: Norma de la velocidad inicial de las partículas.
interaction_strength: Fuerza de interacción entre las partículas.
noise_var: Varianza del ruido añadido a las observaciones.
_charge_types: Tipos de cargas (por ejemplo, -1, 0, 1).
_delta_T: Paso de tiempo para la simulación.
_max_F: Fuerza máxima permitida basada en el paso de tiempo.
Métodos
__init__: Inicializa la simulación con los parámetros dados y configura los tipos de cargas, el paso de tiempo y la fuerza máxima.

_l2: Calcula la distancia L2 (euclidiana) entre cada par de filas en dos matrices. Esto se utiliza para determinar las distancias entre partículas.

_energy: Calcula la energía total del sistema (cinética y potencial) basada en las posiciones y velocidades de las partículas y las conexiones entre ellas. La energía potencial se calcula en función de la ley de Coulomb.

_clamp: Asegura que las partículas permanezcan dentro de los límites de la caja. Ajusta las posiciones y velocidades para simular colisiones elásticas con las paredes de la caja.

sample_trajectory: Genera una trayectoria de muestra para las partículas en la simulación. Este método:

Inicializa las posiciones y velocidades de las partículas.
Simula la dinámica del sistema durante un tiempo T, actualizando las posiciones y velocidades en cada paso de tiempo utilizando el método de leapfrog.
Añade ruido a las observaciones.
Devuelve las posiciones, velocidades y conexiones entre las partículas a lo largo del tiempo.
Utilidad
La clase ChargedParticlesSim es útil para simular sistemas físicos donde las partículas interactúan a través de fuerzas de Coulomb, 
como en modelos de partículas cargadas o sistemas de iones. La simulación puede ser utilizada para estudiar el comportamiento dinámico de estos sistemas y generar datos para entrenar modelos de aprendizaje automático que infieren dinámicas físicas."""