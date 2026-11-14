P_D = 0.01
P_Pos_D = 0.95
P_Pos_NoD = 0.05

P_Pos = (P_D * P_Pos_D) + ((1-P_D) * P_Pos_NoD)

P_D_Pos = (P_D * P_Pos_D) / P_Pos

print("Probability of disease after positive test:",
      round(P_D_Pos, 4))
