import cirq

q = cirq.LineQubit(0)           # 實例化一個量子位元
simulator = cirq.Simulator()    # 實例化一個模擬器

X_circuit = cirq.Circuit(       # 建立一個包含量子NOT閘和測量的量子電路
    cirq.X(q),
    cirq.measure(q)
)
print(X_circuit)                # 在終端可視化輸出量子電路

# 使用模擬器對該量子電路進行20次的模擬測量
result = simulator.run(X_circuit, repetitions=20)
print(result)                   # 輸出模擬測量結果

H_circuit = cirq.Circuit(       # 建立一個包含阿達馬閘和測量的量子電路
    cirq.H(q),
    cirq.measure(q)
)
print(H_circuit)
result = simulator.run(H_circuit, repetitions=20)
print(result)