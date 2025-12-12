# 21111-
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. 기본 설정
# -----------------------------
G = 6.67430e-11  # 중력상수

# 천체 질량 (kg)
m1 = 1.989e30     # 태양 질량
m2 = 5.972e24     # 지구 질량

# 초기 거리 (m)
r_initial = np.array([1.5e11, 0.0])  # 지구 위치 (x축 기준)
v_initial = np.array([0.0, 29780.0]) # 지구 초기 공전 속력 (m/s)

# 태양은 질량 중심 기준 위치 계산
r1 = - (m2 / (m1 + m2)) * r_initial  # 태양 위치
r2 = (m1 / (m1 + m2)) * r_initial    # 지구 위치

v1 = - (m2 / (m1 + m2)) * v_initial  # 태양 초기 속도
v2 = (m1 / (m1 + m2)) * v_initial    # 지구 초기 속도

# 시뮬레이션 설정
dt = 5000         # 시간 간격 (초)
steps = 5000      # 전체 프레임 수

# 결과 저장 리스트
r1_list = []
r2_list = []


# -----------------------------
# 2. Verlet 적분 함수
# -----------------------------
def acceleration(r1, r2):
    r = r2 - r1
    dist = np.linalg.norm(r)
    a1 = G * m2 * r / dist**3
    a2 = -G * m1 * r / dist**3
    return a1, a2


# -----------------------------
# 3. 시뮬레이션
# -----------------------------
for _ in range(steps):
    r1_list.append(r1.copy())
    r2_list.append(r2.copy())

    # 가속도 계산
    a1, a2 = acceleration(r1, r2)

    # Verlet 업데이트
    r1_new = r1 + v1 * dt + 0.5 * a1 * dt**2
    r2_new = r2 + v2 * dt + 0.5 * a2 * dt**2

    a1_new, a2_new = acceleration(r1_new, r2_new)

    v1 = v1 + 0.5 * (a1 + a1_new) * dt
    v2 = v2 + 0.5 * (a2 + a2_new) * dt

    r1, r2 = r1_new, r2_new


# -----------------------------
# 4. 시각화
# -----------------------------
r1_arr = np.array(r1_list)
r2_arr = np.array(r2_list)

plt.figure(figsize=(7,7))
plt.plot(r1_arr[:,0], r1_arr[:,1], label="Star (m1)")
plt.plot(r2_arr[:,0], r2_arr[:,1], label="Planet (m2)")

plt.scatter([0], [0], color='black', s=30)  # 질량 중심 (0,0)

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("Barycenter-based Two-Body Orbit Simulation")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()
