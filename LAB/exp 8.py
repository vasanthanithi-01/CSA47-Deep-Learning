P_A = 0.6
P_B = 0.4

P_DA = 0.02
P_DB = 0.05

P_D = (P_A * P_DA) + (P_B * P_DB)

P_A_D = (P_A * P_DA) / P_D

print("Probability of defective product:", P_D)
print("Probability it came from Machine A:", round(P_A_D, 4))
