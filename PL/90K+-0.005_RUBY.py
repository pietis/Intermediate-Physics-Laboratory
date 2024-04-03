import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import impulse

# ASC 파일에서 데이터를 가져와서 array로 변환하는 함수
def read_asc_file(file_path):
    # ASC 파일의 y data를 array로 반환
    df = pd.read_csv(file_path, sep=';', header=None)
    y_data = []
    for index, row in df.iterrows():
        y_data.append(float(row[1]))
    return y_data

# 1. 노이즈 array 만들기=================================================================================================================================

# 5개의 Noise ASC 파일 경로
noise_paths = ["PL/Data/90K+-0.005_RUBY/90K_RUBY_NOISE1.asc",
               "PL/Data/90K+-0.005_RUBY/90K_RUBY_NOISE2.asc",
               "PL/Data/90K+-0.005_RUBY/90K_RUBY_NOISE3.asc",
               "PL/Data/90K+-0.005_RUBY/90K_RUBY_NOISE4.asc",
               "PL/Data/90K+-0.005_RUBY/90K_RUBY_NOISE5.asc"]

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

# 2. array 만들기=================================================================================================================================

# 5개의 파일 경로
data_paths = ["PL/Data/90K+-0.005_RUBY/90K_RUBY1.asc",
              "PL/Data/90K+-0.005_RUBY/90K_RUBY2.asc",
              "PL/Data/90K+-0.005_RUBY/90K_RUBY3.asc",
              "PL/Data/90K+-0.005_RUBY/90K_RUBY4.asc",
              "PL/Data/90K+-0.005_RUBY/90K_RUBY5.asc"]

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

# 4. Fitting & R^2 값구하기 =================================================================================================================================

x_fit = np.linspace(min(x_data), max(x_data), len(x_data))

'''total_variation = sum((y_data - np.mean(y_data))**2)
residuals = y_data - y_fit
residual_variation = sum(residuals**2)
r_squared = 1 - (residual_variation / total_variation)
print("R squared value is", r_squared)'''

# 5. 그래프 그리기 =================================================================================================================================

plt.figure()
plt.plot(x_data, y_data, label = 'Original data', color = 'black')
'''plt.plot(x_fit, y_fit, 'r-', label = 'Fitted Curve')'''
plt.xlabel('Wavelength[nm]')
plt.ylabel('Photon counts')
plt.legend()
plt.show()
