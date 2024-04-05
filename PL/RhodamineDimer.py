import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ASC 파일에서 데이터를 가져와서 array로 변환하는 함수
def read_asc_file(file_path):
    # ASC 파일의 y data를 array로 반환
    df = pd.read_csv(file_path, sep=';', header=None)
    y_data = []
    for index, row in df.iterrows():
        y_data.append(float(row[1]))
    return y_data

# 1. 노이즈 array 만들기 =================================================================================================================================

# 5개의 Noise ASC 파일 경로
noise_paths = ["PL/Data/NOISEOF567AREA/NOISEOFRODAMINE1.asc",
               "PL/Data/NOISEOF567AREA/NOISEOFRODAMINE2.asc",
               "PL/Data/NOISEOF567AREA/NOISEOFRODAMINE3.asc",
               "PL/Data/NOISEOF567AREA/NOISEOFRODAMINE4.asc",
               "PL/Data/NOISEOF567AREA/NOISEOFRODAMINE5.asc"]

# 각 파일에서 데이터를 가져와 array로 변환하여 리스트에 저장
noise_list = [read_asc_file(noise) for noise in noise_paths]

# x좌표 생성
df = pd.read_csv(noise_paths[0], sep=';', header=None)
x_data = []
for index, row in df.iterrows():
        x_data.append(float(row[0])) 

# float 앞에 1239.8/ 넣으면 x축을 [eV]로 바꾼다.
# 이 데이터는 우리 Photoluminescence 실험 전반에서 같으니 x_data를 계속 사용하자.

# 평균 Noise인 'noise result' array 생성 
noise_result = []
for i in range(len(noise_list[0])):
    a = 0
    for j in range(len(noise_list)):
        a += noise_list[j][i]
    noise_result.append(a/5)

# 2. 로다민 array 만들기 =================================================================================================================================

# 5개의 Rhodamine 파일 경로
data_paths = ["PL/Data/RODAMINE/RODAMINE1.asc",
              "PL/Data/RODAMINE/RODAMINE2.asc",
              "PL/Data/RODAMINE/RODAMINE3.asc",
              "PL/Data/RODAMINE/RODAMINE4.asc",
              "PL/Data/RODAMINE/RODAMINE5.asc"]

# 각 파일에서 데이터를 가져와 array로 변환하여 리스트에 저장
data_list = [read_asc_file(data) for data in data_paths]

data_result = []
for i in range(len(data_list[0])):
    a = 0
    for j in range(len(data_list)):
        a += data_list[j][i]
    data_result.append(a/5)

# 3. 노이즈 제거 하고 y_data 생성 =================================================================================================================================

y_data = []
for i in range(1024):
    y_data.append(data_result[i] - noise_result[i])

# 4. Gaussian Fitting & R^2 값구하기 =================================================================================================================================

# 가우시안 함수 정의
def gaussiandimer(x, amp1, cen1, wid1, amp2, cen2, wid2):
    return amp1 * np.exp(-(x - cen1)**2 / (2 * wid1**2)) + amp2 * np.exp(-(x - cen2)**2 / (2 * wid2**2))

# 초기 추정값 설정
initial_guess = [1600, 560, 10, 1600, 560, 10]

# curve_fit을 사용하여 가우시안 함수 파라미터 추정
popt, pcov = curve_fit(gaussiandimer, x_data, y_data, p0=initial_guess)

# 추정된 파라미터
amp1, cen1, wid1, amp2, cen2, wid2 = popt

# 추정된 파라미터를 사용하여 가우시안 함수 적용
y_fit = gaussiandimer(x_data, amp1, cen1, wid1, amp2, cen2, wid2)

total_variation = sum((y_data - np.mean(y_data))**2)
residuals = y_data - y_fit
residual_variation = sum(residuals**2)
r_squared = 1 - (residual_variation / total_variation)
print("R squared value is", r_squared)

print(popt, pcov)

# 5. 그래프 그리기 =================================================================================================================================

plt.figure()
plt.plot(x_data, y_data, 'k-', label = 'Original data')
plt.plot(x_data, y_fit, 'r-', label = 'Gaussian curve')
plt.xlabel('Wavelength[nm]')
plt.ylabel('Counts')
plt.legend()
plt.show()