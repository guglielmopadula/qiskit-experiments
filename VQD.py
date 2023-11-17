from qiskit.circuit.library import TwoLocal

ansatz = TwoLocal(2, rotation_blocks=["ry", "rz"], entanglement_blocks="cz", reps=1)

ansatz.decompose().draw('mpl')
import numpy as np

from qiskit.circuit import ParameterVector

def calculate_overlap(circuits_1, circuits_2, parameters, sampler):

    def get_global_fidelity(probability_distribution):
        """Process the probability distribution of a measurement to determine the
        global fidelity.
        """
        return probability_distribution.get(0, 0)
    
    def create_fidelity_circuit(circuit_1, circuit_2):
        if len(circuit_1.clbits) > 0:
            circuit_1.remove_final_measurements()
        if len(circuit_2.clbits) > 0:
            circuit_2.remove_final_measurements()

        circuit = circuit_1.compose(circuit_2.inverse())
        circuit.measure_all()
        return circuit
    
    def construct_fidelity_circuits(circuits_1, circuits_2):
        """
        Constructs the list of fidelity circuits to be evaluated.
        These circuits represent the state overlap between pairs of input circuits,
        and their construction depends on the fidelity method implementations.
        """
        circuits = []
        for (circuit_1, circuit_2) in zip(circuits_1, circuits_2):
            # re-parametrize input circuits
            parameters_1 = ParameterVector("x", circuit_1.num_parameters)
            parametrized_circuit_1 = circuit_1.assign_parameters(parameters_1)
            parameters_2 = ParameterVector("y", circuit_2.num_parameters)
            parametrized_circuit_2 = circuit_2.assign_parameters(parameters_2)

            circuit = create_fidelity_circuit(
                parametrized_circuit_1, parametrized_circuit_2
            )
            circuits.append(circuit)

        return circuits

    circuits = construct_fidelity_circuits(circuits_1, circuits_2)

    job = sampler.run(circuits=circuits, parameter_values=parameters)
    result = job.result()

    raw_fidelities = [
        get_global_fidelity(prob_dist) for prob_dist in result.quasi_dists
    ]

    return np.clip(raw_fidelities, 0, 1).tolist()

from qiskit.primitives import Sampler, Estimator
estimator=Estimator()
sampler = Sampler()

def cost_func_vqd(parameters, ansatz, prev_states, step, betas, estimator, sampler, hamiltonian):
    
    if len(parameters.shape) == 1:
        parameters = np.reshape(parameters, (-1, ansatz.num_parameters))

    batch_size = len(parameters)

    estimator_job = estimator.run(
        batch_size * [ansatz], batch_size * [hamiltonian], parameters
    )

    total_cost = np.zeros(batch_size)

    if step > 1:
        # compute overlap cost
        batched_prev_states = [state for state in prev_states for _ in range(batch_size)]
        costs = calculate_overlap(
            batch_size * [ansatz] * (step - 1),
            batched_prev_states,
            np.tile(parameters, (step - 1, 1)),
            sampler
        )

        costs = np.reshape(costs, (step - 1, -1))
        for state, cost in enumerate(costs):
            total_cost += np.real(betas[state] * cost)

    estimator_result = estimator_job.result()

    values = estimator_result.values + total_cost

    return values if len(values) > 1 else values[0]

from qiskit.quantum_info import SparsePauliOp

k = 3
betas = [33, 33, 33]
x0 = np.zeros(8)
observable = SparsePauliOp.from_list([("II", 2), ("XX", -2), ("YY", 3), ("ZZ", -3)])

prev_states = []
prev_opt_parameters = []
eigenvalues = []


from scipy.optimize import minimize

for step in range(1, k + 1):
    
    if step > 1:
        prev_states.append(ansatz.bind_parameters(prev_opt_parameters))
    
    result = minimize(cost_func_vqd, x0, args=(ansatz, prev_states, step, betas, estimator, sampler, observable), method="COBYLA")
    print(result)

    prev_opt_parameters = result.x
    eigenvalues.append(result.fun)

print(eigenvalues)