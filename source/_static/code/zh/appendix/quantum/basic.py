import cirq

q = cirq.LineQubit(0)           # 实例化一个量子比特
simulator = cirq.Simulator()    # 实例化一个模拟器

X_circuit = cirq.Circuit(       # 建立一个包含量子非门和测量的量子线路
    cirq.X(q),
    cirq.measure(q)
)
print(X_circuit)                # 在终端可视化输出量子线路

# 使用模拟器对该量子线路进行20次的模拟测量
result = simulator.run(X_circuit, repetitions=20)
print(result)                   # 输出模拟测量结果

H_circuit = cirq.Circuit(       # 建立一个包含阿达马门和测量的量子线路
    cirq.H(q),
    cirq.measure(q)
)
print(H_circuit)
result = simulator.run(H_circuit, repetitions=20)
print(result)