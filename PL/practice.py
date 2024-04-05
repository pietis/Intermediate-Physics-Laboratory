import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.optimize import curve_fit

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
noise_paths = ["PL/Data/NOISEWITHOUTHAND/10K_NoiseWithoutHand1.asc",
               "PL/Data/NOISEWITHOUTHAND/10K_NoiseWithoutHand2.asc",
               "PL/Data/NOISEWITHOUTHAND/10K_NoiseWithoutHand3.asc",
               "PL/Data/NOISEWITHOUTHAND/10K_NoiseWithoutHand4.asc",
               "PL/Data/NOISEWITHOUTHAND/10K_NoiseWithoutHand5.asc"]

# 각 파일에서 데이터를 가져와 array로 변환하여 리스트에 저장
noise_list = [read_asc_file(noise) for noise in noise_paths]

# x좌표 생성
df = pd.read_csv(noise_paths[0], sep=';', header=None)
x_data = []
for index, row in df.iterrows():
        x_data.append(1239.8/float(row[0]))

# float 앞에 '1239.8/'을 넣으면 x축을 [eV]로 바꾼다.

# 평균 Noise인 'noise result' array 생성 
noise_result = []
for i in range(len(noise_list[0])):
    a = 0
    for j in range(len(noise_list)):
        a += noise_list[j][i]
    noise_result.append(a/5)

# 2. array 만들기=================================================================================================================================

# 5개의 파일 경로
data_paths = ["PL/Data/10K+-0.2_RUBY/10K_RUBY1.asc",
              "PL/Data/10K+-0.2_RUBY/10K_RUBY2.asc",
              "PL/Data/10K+-0.2_RUBY/10K_RUBY3.asc",
              "PL/Data/10K+-0.2_RUBY/10K_RUBY4.asc",
              "PL/Data/10K+-0.2_RUBY/10K_RUBY5.asc"]

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
y_rel_data = []
for i in range(1024):
    y_data.append(data_result[i] - noise_result[i])

for i in y_data:
    y_rel_data.append(i/max(y_data))

x_data = x_data[::-1] # 쉬운 데이터 처리를 위해 데이터 뒤집음
y_data = y_data[::-1]
y_rel_data = y_rel_data[::-1]

# 4.  Fitting 값 구하기 =================================================================================================================================

# Gaussian 함수 정의
def gaussian(x, x0, sigma0, x1, sigma1):
    return np.exp(-(x - x0)**2 / (2 * sigma0**2)) / (np.sqrt(2 * np.pi) * sigma0) + np.exp(-(x - x1)**2 / (2 * sigma1**2)) / (np.sqrt(2 * np.pi) * sigma1)

initial_guess = [1.7898644299684574, 0.00018940549781398808, 1.7936623131076712, 0.0037036715120915]

gaussian_func, pcov = curve_fit(gaussian, x_data, y_rel_data, p0 = initial_guess)

x0, sigma0, x1, sigma1 = gaussian_func

print(x0, sigma0, x1, sigma1)
y_fit = gaussian(x_data, x0, sigma0, x1, sigma1)

plt.figure()
plt.plot(x_data, y_rel_data, 'k-', label = 'Original data')
plt.plot(x_data, y_fit)
plt.xlabel('Energy [eV]')
plt.ylabel('Photon counts')
plt.legend()
plt.show()

'''
# Lorentzian 함수 정의
def lorentzian(x, x0, gamma):
    return (1 / np.pi) * (gamma / ((x - x0)**2 + (0.5 * gamma)**2))

dx = x_data[1] - x_data[0]  # 배열 간격

# Gaussian 및 Lorentzian 함수 생성
gaussian_func = gaussian(np.array(x_data), x0 = 1, sigma=0.01)
lorentzian_func = lorentzian(np.array(x_data), x0 = 2, gamma=0.01)

# Gaussian 함수와 Lorentzian 함수의 컨볼루션
convolution_result = convolve(gaussian_func, lorentzian_func, mode='same') * dx

print(max(gaussian_func))


# 5. 그래프 그리기 =================================================================================================================================
plt.figure()
plt.plot(x_data, y_data, 'k-', label = 'Original data')
plt.plot(x_data, convolution_result)
plt.xlabel('Energy [eV]')
plt.ylabel('Photon counts')
plt.legend()
plt.show()'''