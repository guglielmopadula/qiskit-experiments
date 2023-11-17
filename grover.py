
from qiskit import QuantumCircuit
from qiskit.algorithms import Grover, AmplificationProblem
from qiskit.primitives import Sampler

good_state = ['11']

# specify the oracle that marks the state '11' as a good solution
oracle = QuantumCircuit(2)
oracle.cz(0, 1)
problem = AmplificationProblem(oracle, is_good_state=good_state)
problem.grover_operator.decompose().draw(output='mpl')
grover = Grover(sampler=Sampler())
result = grover.amplify(problem)
print('Result type:', type(result))
print()
print('Success!' if result.oracle_evaluation else 'Failure!')
print('Top measurement:', result.top_measurement)

