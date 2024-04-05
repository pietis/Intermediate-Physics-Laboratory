import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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
noise_paths = ["PL/Data/290K+-0.03_RUBY_46mTorr/290K_RUBY_NOISE1.asc",
               "PL/Data/290K+-0.03_RUBY_46mTorr/290K_RUBY_NOISE2.asc",
               "PL/Data/290K+-0.03_RUBY_46mTorr/290K_RUBY_NOISE3.asc",
               "PL/Data/290K+-0.03_RUBY_46mTorr/290K_RUBY_NOISE4.asc",
               "PL/Data/290K+-0.03_RUBY_46mTorr/290K_RUBY_NOISE5.asc"]

# 각 파일에서 데이터를 가져와 array로 변환하여 리스트에 저장
noise_list = [read_asc_file(noise) for noise in noise_paths]

# x좌표 생성
df = pd.read_csv(noise_paths[0], sep=';', header=None)
x_data = []
for index, row in df.iterrows():
        x_data.append(1239.8/float(row[0]))

# float 앞에 1239.8/ 넣으면 x축을 [eV]로 바꾼다.

# 평균 Noise인 'noise result' array 생성 
noise_result = []
for i in range(len(noise_list[0])):
    a = 0
    for j in range(len(noise_list)):
        a += noise_list[j][i]
    noise_result.append(a/5)

# 2. array 만들기=================================================================================================================================

# 5개의 파일 경로
data_paths = ["PL/Data/290K+-0.03_RUBY_46mTorr/290K_RUBY_1.asc",
              "PL/Data/290K+-0.03_RUBY_46mTorr/290K_RUBY_2.asc",
              "PL/Data/290K+-0.03_RUBY_46mTorr/290K_RUBY_3.asc",
              "PL/Data/290K+-0.03_RUBY_46mTorr/290K_RUBY_4.asc",
              "PL/Data/290K+-0.03_RUBY_46mTorr/290K_RUBY_5.asc"]

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

x_data = x_data[::-1]
y_data = y_data[::-1]

# 4. Fitting & R^2 값구하기 =================================================================================================================================

# 5. 그래프 그리기 =================================================================================================================================

plt.figure()
plt.ylim(0, 360)
plt.plot(x_data, y_data, 'k-', label = 'Original data')
plt.xlabel('Energy [eV]')
plt.ylabel('Photon counts')
plt.legend()
plt.show()

# 6.Peak position, height, FWHM 찾기

def find_peak_properties(x, y):
    x = np.array(x)  # 리스트를 NumPy 배열로 변환
    y = np.array(y)  # 리스트를 NumPy 배열로 변환
    
    # Peak 찾기
    peaks, _ = find_peaks(y, height=1000)  # 높이가 1000 이상인 모든 peak를 찾기

    peak_properties = []
    for peak_index in peaks:
        peak_x = x[peak_index]  # peak의 x 좌표
        peak_y = y[peak_index]  # peak의 y 좌표

        # 반치적폭 찾기
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
