import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import voigt_profile
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# ASC 파일에서 데이터를 가져와서 array로 변환하는 함수
def read_asc_file(file_path):
    # ASC 파일의 y data를 array로 반환
    df = pd.read_csv(file_path, sep=';', header=None)
    y = []
    for index, row in df.iterrows():
        y.append(float(row[1]))
    return y

# 1. 노이즈 array 만들기=================================================================================================================================

# 5개의 Noise ASC 파일 경로
noise_paths = ["PL/Data/250K+-0.01_RUBY_7.4mTorr/250K_RUBY_NOISE%d.asc"%(i) for i in range(1,6)]

# 각 파일에서 데이터를 가져와 array로 변환하여 리스트에 저장
noise_list = [read_asc_file(noise) for noise in noise_paths]

df = pd.read_csv(noise_paths[0], sep=';', header=None)
x_data = []
for index, row in df.iterrows():
        x_data.append(1239.8/float(row[0]))
# float 앞에 '1239.8/'을 넣으면 x축을 [eV]로 바꾼다.

# 평균 Noise인 'noise result' array 생성
noise_result = []
noise_result = np.sum(noise_list, axis=0)/5

# 2. array 만들기=================================================================================================================================

# 5개의 파일 경로
data_paths = ["PL/Data/250K+-0.01_RUBY_7.4mTorr/250K_RUBY_%d.asc"%(i) for i in range(1,6)]

# 각 파일에서 데이터를 가져와 array로 변환하여 리스트에 저장
data_list = [read_asc_file(data) for data in data_paths]
data_result = np.sum(data_list, axis=0)/5

# 3. 노이즈 제거 하고 y_data 생성 =================================================================================================================================

y_data = []
y_data = np.array(data_result) - np.array(noise_result)

x_data = x_data[::-1] # 쉬운 데이터 처리를 위해 데이터 뒤집음
y_data = y_data[::-1]

# 4. Fitting =================================================================================================================================

# Gaussian 함수 정의
def gaussian(x, amp0, x0, sigma0, amp1, x1, sigma1):
    return amp0 * np.exp(-(x - x0)**2 / (2 * sigma0**2)) / (np.sqrt(2 * np.pi) * sigma0) + amp1 * np.exp(-(x - x1)**2 / (2 * sigma1**2)) / (np.sqrt(2 * np.pi) * sigma1)
2
def lorentzian(x, amp0, x0, gamma0, amp1, x1, gamma1):
    return amp0*(1 / np.pi) * (gamma0 / ((x - x0)**2 + (0.5 * gamma0)**2)) + amp1*(1 / np.pi) * (gamma1 / ((x - x1)**2 + (0.5 * gamma1)**2))

def voigt(x, amp0, x0, sigma0, gamma0, amp1, x1, sigma1, gamma1):
    return amp0*voigt_profile(x-x0, sigma0, gamma0) + amp1*voigt_profile(x-x1, sigma1, gamma1)

# trunc = [1.78 < x < 1.80 for x in x_data]
# x_data = np.array(x_data)[trunc]
# y_data = np.array(y_data)[trunc]

initial_guess = [5000, 1.7888236411209433, 0.00018940549781398808, 0.00018940549781398808, 
                 4450, 1.7926158598731354, 0.000037036715120915, 0.000037036715120915] 
# fitting이 잘 안되면 열심히 조절할 것

voigt_func, pcov = curve_fit(voigt, x_data, y_data, p0 = initial_guess)

print(np.sqrt(np.diag(pcov)))

y_fit = voigt(x_data, *voigt_func)

def find_r_squared(y_data, y_fit):
    total_variation = sum((y_data - np.mean(y_data))**2)
    residuals = y_data - y_fit
    residual_variation = sum(residuals**2)
    r_squared = 1 - (residual_variation / total_variation)
    print("R squared value:", r_squared)

find_r_squared(y_data, y_fit)

# 5. Peak position, height, FWHM 찾기 =================================================================================================================================

def find_peak_properties(x, y):
    x = np.array(x)  # 리스트를 NumPy 배열로 변환
    y = np.array(y)  # 리스트를 NumPy 배열로 변환
    
    # Peak 찾기
    peaks, _ = find_peaks(y, height=1000)  # 높이가 1000 이상인 모든 peak를 찾기

    peak_properties = []
    for peak_index in peaks:
        peak_x = x[peak_index]  # peak의 x 좌표
        peak_y = y[peak_index]  # peak의 y 좌표

        # FWHM 찾기
        half_max_height = peak_y / 2
        left_index = np.argmin(np.abs(y[:peak_index] - half_max_height))
        right_index = np.argmin(np.abs(y[peak_index:] - half_max_height)) + peak_index
        fwhm = x[right_index] - x[left_index]

        # 결과 저장
        peak_properties.append({
            'x': peak_x,
            'y': peak_y,
            'fwhm': fwhm
        })

    return peak_properties

# Peak 속성 찾기
peak_properties = find_peak_properties(x_data, y_data)

# 결과 출력
for i, peak in enumerate(peak_properties, 1):
    print(f"Peak {i}:")
    print(f"   X-coordinate: {peak['x']}")
    print(f"   Y-coordinate: {peak['y']}")
    print(f"   FWHM: {peak['fwhm']}")

# 6. 그래프 그리기 =================================================================================================================================

plt.figure()
plt.plot(x_data, y_data, 'k-', label = 'Original data')
plt.plot(x_data, y_fit)
plt.xlabel('Energy [eV]')
plt.ylabel('Counts')
plt.legend()
plt.show()