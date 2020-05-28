import cirq

q_0, q_1 = cirq.LineQubit(0), cirq.LineQubit(1)

circuit = cirq.Circuit(
    cirq.H(q_0),
    cirq.X(q_1),
    cirq.CNOT(q_0, q_1),
    cirq.measure(q_0, key='q_0'),
    cirq.measure(q_1, key='q_1')
)
print(circuit)

simulator = cirq.Simulator()
result = simulator.run(circuit, repetitions=20)
print(result)